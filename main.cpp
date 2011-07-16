/***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

/* author:  zhujiang  (email: zhujiangmail@hotmail.com , zhujiang@vmail.btvnet.net) */

#include <cv.h>
#include <highgui.h>

#include <math.h>
#include <iostream>
#include <omp.h>

#define  ZGDN          1
#define  ZGDIRES       32

#define  SCANMXYNI     6
#define  SCANMXYNF     16

#define  INMAXNZP      6
#define  FIMAXNZP      80

#define  SCANSTEP      4

#define  TRNOISE       50.0
#define  SCNOISE       20.0

#define  INZGTHRE      0.7
#define  FIZGTHRE      0.75

#define  TRIFTH      4.0
#define  SCIFTH      3.0

#define  NUMTHS      8

struct  ScanDire
{
        int  x1;
        int  y1;
        int  x2;
        int  y2;
};

struct ScanPoint
{
    int  x;
    int  y;
    int  idx;
    int   na;
    float   zoom;
    float   va;
    float   similarity;
};

struct ZPoint
{
    int  x;
    int  y;
    int  xr;
    int  yr;
    int  dire;
    float   r;
    float   a;
    int  mx;
    int  my;
};

struct ZGFeature
{
    float  dire_ves[ZGDIRES];
    float  ve_average;
    float  max_ve;
    int    num_dires[ZGDIRES];
    int    mask_dires[ZGDIRES];
    int    mask_mdires[ZGDIRES];
    int    mask_noise;
};

ScanDire   scan_dires[32];

CvPoint    scan_imxys[SCANMXYNI+1];
CvPoint    scan_fmxys[SCANMXYNF+1];

class  ZeGabor
{
    public:
    ZeGabor(double dPhi, int  Nu, double dSigma, double dF);
    ~ZeGabor();
    void   conv_mat(CvMat *src_mat, CvMat *dst_mat);
    CvMat* get_kreal() {return  Real;};
    CvMat* get_kimag() {return  Imag;};

    protected:
    long    Width;
    CvMat   *Imag;
    CvMat   *Real;
};

ZeGabor::ZeGabor(double dPhi, int  Nu, double dSigma, double dF)
{
        double dNu = Nu;
    double Sigma = dSigma;
    double F = dF;

    double Kmax = CV_PI/2;

    // Absolute value of K
    double K = Kmax / pow(F, dNu);
    double Phi = dPhi;

        double dModSigma = Sigma/K;

        double dWidth = cvRound(dModSigma*6 + 1);

        if (fmod(dWidth, 2.0)==0.0) dWidth++;

    Width = (long)dWidth;

    Real = cvCreateMat( Width, Width, CV_32FC1);
    Imag = cvCreateMat( Width, Width, CV_32FC1);

        CvMat *mReal, *mImag;
        mReal = cvCreateMat( Width, Width, CV_32FC1);
        mImag = cvCreateMat( Width, Width, CV_32FC1);

        double a, b;
        double c;
        double x, y;
        double ra, rb;
        double dReal;
        double dImag;
        double dTemp1, dTemp2, dTemp3;

        a = 0.425;
        b = 1.0;
        c = dPhi;

        for (int i = 0; i < Width; i++)
        {
            for (int j = 0; j < Width; j++)
            {
                x = i-(Width-1)/2;
                y = j-(Width-1)/2;

                ra = x*cos(c) + y*sin(c);
                rb = -x*sin(c) + y*cos(c);

                dTemp1 = (K*K/Sigma*Sigma)*exp(-(ra*ra/(a*a)+rb*rb/(b*b))*K*K/(2*Sigma*Sigma));
                dTemp2 = cos(K*cos(Phi)*1.25*x + K*sin(Phi)*1.25*y) - exp(-(pow(Sigma,2)/2));
                dTemp3 = sin(K*cos(Phi)*1.25*x + K*sin(Phi)*1.25*y);


                dReal = dTemp1*dTemp2;
                dImag = dTemp1*dTemp3;

                cvSetReal2D((CvMat*)mReal, i, j, dReal );
                cvSetReal2D((CvMat*)mImag, i, j, dImag );
            }
        }
        /**************************** Gabor Function ****************************/
        cvCopy(mReal, Real, NULL);
        cvCopy(mImag, Imag, NULL);
        //printf("A %d x %d Gabor kernel with %f PI in arc is created.\n", Width, Width, Phi/PI);
        cvReleaseMat( &mReal );
        cvReleaseMat( &mImag );
}

void   ZeGabor::conv_mat(CvMat *src_mat, CvMat *dst_mat)
{
    CvMat *mat = cvCreateMat(src_mat->height, src_mat->width, CV_32FC1);

    cvFilter2D( (CvMat*)src_mat, (CvMat*)mat, (CvMat*)Imag, cvPoint( (Width-1)/2, (Width-1)/2));

    cvPow(mat,mat,2);

    cvPow(mat,dst_mat,0.5);

    cvReleaseMat( &mat );
}

ZeGabor::~ZeGabor()
{
    cvReleaseMat( &Real );
    cvReleaseMat( &Imag );
}

void  train_zfilters(IplImage  *img_gray, std::vector<ZPoint>   &filter_zpoints)
{
    CvMat     *mat_gray = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);
        CvMat     *mat_dire = cvCreateMat(img_gray->height, img_gray->width, CV_32SC1);
        CvMat     *mat_mask = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);
        CvMat     *mat_mva = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);
        CvMat     *mat_max_ve = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);

        cvConvert(img_gray, mat_gray);

        CvMat  *mat_mags[ZGDIRES];

        for (int i=0; i<ZGDIRES; i++)
        {
                mat_mags[i] = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);
        }

        ZeGabor   *gabors[ZGDIRES];

        double  Sigma = 2*CV_PI;
        double  F = sqrt(2.0);
        double  dn = ZGDN;

        //#pragma omp parallel for

        for (int n=0; n<ZGDIRES; n++)
        {
                gabors[n] = new ZeGabor((CV_PI/ZGDIRES)*n, dn, Sigma, F);
        }

        //#pragma omp parallel for

        for (int n=0; n<ZGDIRES; n++)
        {
                gabors[n]->conv_mat(mat_gray, mat_mags[n]);
        }

        //#pragma omp parallel for

        float xr, yr;

        int  xz = img_gray->width/2;
        int  yz = img_gray->height/2;

        for (int y=0; y<img_gray->height; y++)
        {
                for (int x=0; x<img_gray->width; x++)
                {
                        float  sum;
                        float  average;
                        float  max_ve;
                        int    dire;

                        sum = 0.0;
                        dire = 0;
                        max_ve = CV_MAT_ELEM(*mat_mags[0], float, y, x);

                        for (int n=0; n<ZGDIRES; n++)
                        {
                                float  ve = CV_MAT_ELEM(*mat_mags[n], float, y, x);
                                sum += ve;
                                if (ve > max_ve)
                                {
                                        max_ve = ve;
                                        dire = n;
                                }
                        }

                        average = sum/ZGDIRES;

                        CV_MAT_ELEM(*mat_mva, float, y, x) = max_ve/average;

                        CV_MAT_ELEM(*mat_max_ve, float, y, x) = max_ve;

                        CV_MAT_ELEM(*mat_dire, int, y, x) = dire;
                }
        }

        cvZero(mat_mask);

        //#pragma omp parallel for

        for (int y=5; y<img_gray->height-5; y+=2)
        {
                for (int x=5; x<img_gray->width-5; x+=2)
                {
                        CV_MAT_ELEM(*mat_mask, float, y, x) = 0.0;

                        int  max_x = x;
                        int  max_y = y;

                        float max_ve = CV_MAT_ELEM(*mat_max_ve, float, y, x);

                        int  mxy = 5;

                        for (int my=-mxy; my<=mxy; my++)
                        {
                                for (int mx=-mxy; mx<=mxy; mx++)
                                {
                                        int y1=y+my;
                                        int x1=x+mx;

                                        float  ve = CV_MAT_ELEM(*mat_max_ve, float, y1, x1);

                                        if (ve > max_ve)
                                        {
                                                max_x = x1;
                                                max_y = y1;

                                                max_ve = ve;
                                        }
                                }
                        }

                        if (max_ve > TRNOISE)
                        {
                                CV_MAT_ELEM(*mat_mask, float, max_y, max_x) = 255.0;
                        }
                }
        }

        //#pragma omp parallel for

        for (int y=5; y<img_gray->height-5; y+=1)
        {
                for (int x=5; x<img_gray->width-5; x+=1)
                {
                        if (CV_MAT_ELEM(*mat_mask, float, y, x) > 254
                            && CV_MAT_ELEM(*mat_mva, float, y, x) > TRIFTH)
                        {
                                ZPoint  zpoint;

                                zpoint.x = x;
                                zpoint.y = y;

                                xr = x - xz;
                                yr = yz - y;

                                float r = sqrt(xr*xr + yr*yr);

                                float a = atan2(yr, xr);

                                if (xr < 0.0001 && xr > -0.0001)
                                {
                                        if (yr > 0) a = 90/360.0*(2*CV_PI);
                                        if (yr < 0) a = 270/360.0*(2*CV_PI);
                                }

                                zpoint.r = r;
                                zpoint.a = a;

                                zpoint.dire = CV_MAT_ELEM(*mat_dire, int, y, x);

                                filter_zpoints.push_back(zpoint);
                        }
                }
        }

        cvReleaseMat(&mat_gray);
        cvReleaseMat(&mat_dire);
        cvReleaseMat(&mat_mask);
        cvReleaseMat(&mat_mva);
        cvReleaseMat(&mat_max_ve);

        for (int n=0; n<ZGDIRES; n++)
        {
                delete gabors[n];
                cvReleaseMat(&mat_mags[n]);
        }
}

void  classification_whirl_zoom(IplImage  *img_templet, IplImage  *img_src, ZGFeature  *local_features, int  nangle,
                                double  zoom, std::vector<ZPoint>  &ifilters,  std::vector<ZPoint>  &ofilter, double  difth01,
                                double  difth02, int  nt_math)
{
        int num_zpoints = ifilters.size();

        //std::cout << "ifilters.size = " << num_zpoints << std::endl;

        ScanPoint  scan_point;

        std::vector<ZPoint>   azfilters;

        azfilters.clear();

        for (std::vector<ZPoint>::iterator  i = ifilters.begin();  i != ifilters.end();  i++)
        {
                ZPoint       zfilter;

                zfilter.a = i->a + (double)nangle/(double)ZGDIRES*CV_PI;
                zfilter.dire = i->dire + nangle;
                zfilter.r = i->r*zoom;

                double  dxr = i->r*zoom*cos(zfilter.a);
                double  dyr = i->r*zoom*sin(zfilter.a);

                zfilter.xr = dxr + 0.5;
                zfilter.yr = dyr + 0.5;

                azfilters.push_back(zfilter);
        }

        int  xz = img_src->width/2;
        int  yz = img_src->height/2;

        double  difths[3];

        difths[0] = difth01;
        difths[1] = difth02;
        difths[2] = difth02;

        for (std::vector<ZPoint>::iterator  i = azfilters.begin();  i != azfilters.end();  i++)
        {
                int xdst = xz + i->xr;
                int ydst = yz - i->yr;

                if (xdst <= 10 || ydst <= 10 || xdst >= img_src->width-10 || ydst >= img_src->height-10)  continue;

                i->x = xdst;
                i->y = ydst;

                int  num_match;

                int dire = i->dire;

                if (dire < 0) dire = dire + ZGDIRES;

                dire = dire%ZGDIRES;

                int mxs[3];
                int mys[3];

                mxs[0] = 0;
                mys[0] = 0;

                mxs[1] = scan_dires[dire].x1;
                mys[1] = scan_dires[dire].y1;

                mxs[2] = scan_dires[dire].x2;
                mys[2] = scan_dires[dire].y2;

                num_match = 0;

                for (int n=0; n<3; n++)
                {
                        int x1 = xdst + mxs[n];
                        int y1 = ydst + mys[n];

                        ZGFeature  *p_features = local_features + y1*img_src->width + x1;

                        if (p_features->max_ve > TRNOISE && p_features->dire_ves[dire] > difths[n]*p_features->ve_average)
                        {
                                num_match++;
                        }

                        if(num_match >= nt_math)
                        {
                                 ofilter.push_back(*i);
                                 break;
                        }
                }
        }
}

void  classification_move_whirl_zoom(IplImage  *img_templet, IplImage  *img_src, ZGFeature  *local_features,
                                int mx, int my, int  nangle, double  zoom, std::vector<ZPoint>  &ifilters,  std::vector<ZPoint>  &ofilter,
                                double  difth01, double  difth02, int  nt_math, double  noise, int  scanmxy)
{
        CvPoint   scan_mxys[scanmxy+1];

        scan_mxys[0].x = 0;
        scan_mxys[0].y = 0;

        for (int n = 1; n <= scanmxy; n++)
        {
                int     x,y;

                double  r = SCANSTEP;

                double  dx = cos(2.0*CV_PI/scanmxy*n)*r;
                double  dy = sin(2.0*CV_PI/scanmxy*n)*r;

                if (dx > 0.0) x = (int)(dx+0.5); else x = (int)(dx-0.5);
                if (dy > 0.0) y = (int)(dy+0.5); else y = (int)(dy-0.5);

                scan_mxys[n].x = x;
                scan_mxys[n].y = y;
        }

        int num_zpoints = ifilters.size();

        //std::cout << "ifilters.size = " << num_zpoints << std::endl;

        ScanPoint  scan_point;

        std::vector<ZPoint>   azfilters;

        azfilters.clear();

        for (std::vector<ZPoint>::iterator  i = ifilters.begin();  i != ifilters.end();  i++)
        {
                ZPoint       zfilter;

                zfilter.a = i->a + (double)nangle/(double)ZGDIRES*CV_PI;
                zfilter.dire = i->dire + nangle;
                zfilter.r = i->r*zoom;

                double  dxr = i->r*zoom*cos(zfilter.a);
                double  dyr = i->r*zoom*sin(zfilter.a);

                zfilter.xr = dxr + 0.5;
                zfilter.yr = dyr + 0.5;

                azfilters.push_back(zfilter);
        }

        int  xz = img_src->width/2 + mx;
        int  yz = img_src->height/2 + my;

        double  difths[3];

        difths[0] = difth01;
        difths[1] = difth02;
        difths[2] = difth02;

        for (std::vector<ZPoint>::iterator  i = azfilters.begin();  i != azfilters.end();  i++)
        {
                int xdst = xz + i->xr;
                int ydst = yz - i->yr;

                if (xdst <= 10 || ydst <= 10 || xdst >= img_src->width-10 || ydst >= img_src->height-10)  continue;

                int dire = i->dire;

                if (dire < 0) dire = dire + ZGDIRES;

                dire = dire%ZGDIRES;

                int mxs[3];
                int mys[3];

                mxs[0] = 0;
                mys[0] = 0;

                mxs[1] = scan_dires[dire].x1;
                mys[1] = scan_dires[dire].y1;

                mxs[2] = scan_dires[dire].x2;
                mys[2] = scan_dires[dire].y2;

                int find;

                find = 0;

                for (int n=0; n<=scanmxy; n++)
                {
                        int mx1 = scan_mxys[n].x;
                        int my1 = scan_mxys[n].y;

                        i->x = xdst;
                        i->y = ydst;

                        int  num_match;

                        num_match = 0;

                        for (int n=0; n<3; n++)
                        {
                                int x1 = xdst + mx1 + mxs[n];
                                int y1 = ydst + my1 + mys[n];

                                ZGFeature  *p_features = local_features + y1*img_src->width + x1;

                                if (p_features->max_ve > noise && p_features->dire_ves[dire] > difths[n]*p_features->ve_average)
                                {
                                        num_match++;
                                }

                                if(num_match >= nt_math)
                                {
                                        find = 1;
                                        break;
                                }
                        }

                        if (find == 1)
                        {
                                ofilter.push_back(*i);
                                break;
                        }
                }
        }
}

void  zfilter_revision(std::vector<ZPoint>   &zfilter,  std::vector<ZPoint>   &zfilterwei, int  maxnzp)
{
        std::vector<ZPoint>   zdirefilters[ZGDIRES];

        int num_zpoint = zfilter.size();

        int num_zpdires[ZGDIRES];

        for (int i=0; i<ZGDIRES; i++)
        {
               zdirefilters[i].clear();
               num_zpdires[i] = 0;
        }

        for (int i=0; i<num_zpoint; i++)
        {
                int n = zfilter[i].dire;
                num_zpdires[n]++;
                zdirefilters[n].push_back(zfilter[i]);
        }

        int  nd = 0;

        for (int i=0; i<ZGDIRES; i++)
        {
                int zf_size = zdirefilters[i].size();

                if (zf_size > num_zpoint/64) nd++;
        }

        int num_ave = num_zpoint/nd;

        //std::cout << "nd = " << nd << std::endl;

        std::vector<ZPoint>   zfilter_tmp;

        for (int i=0; i<ZGDIRES; i++)
        {
                int zf_size = zdirefilters[i].size();

                if (zf_size == 0) continue;

                //int  step = 1;

                int step = zf_size/num_ave + 1;

                //if (step == 0) step = 1;

                for (int l=0; l<zf_size; l+=step)
                {
                        zfilter_tmp.push_back(zdirefilters[i][l]);
                }
        }

        num_zpoint = zfilter_tmp.size();

        for (int i=0; i<ZGDIRES; i++)
        {
               num_zpdires[i] = 0;
        }

        for (int i=0; i<num_zpoint; i++)
        {
                int n = zfilter_tmp[i].dire;
                num_zpdires[n]++;
        }

        int zstep = 1;

        if (num_zpoint > maxnzp) zstep = num_zpoint/maxnzp + 1;

        for (int i=0; i<num_zpoint; i+=zstep)
        {
              zfilterwei.push_back(zfilter_tmp[i]);
        }
}

void  image_local_features(IplImage  *img_src, ZGFeature  *local_features)
{
        IplImage   *img_gray = cvCreateImage( cvGetSize(img_src), 8, 1 );
        cvCvtColor(img_src, img_gray, CV_BGR2GRAY);

    CvMat     *mat_gray = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);

        cvConvert(img_gray, mat_gray);

        CvMat     *mat_edge = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);
        CvMat     *mat_dire = cvCreateMat(img_gray->height, img_gray->width, CV_32SC1);
        CvMat     *mat_mask = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);

        cvZero(mat_edge);
        cvZero(mat_dire);
        cvZero(mat_mask);

        CvMat  *mat_mags[ZGDIRES];

        for (int i=0; i<ZGDIRES; i++)
        {
                mat_mags[i] = cvCreateMat(img_gray->height, img_gray->width, CV_32FC1);
        }

        ZeGabor   *gabors[ZGDIRES];

        double  Sigma = 2*CV_PI;
        double  F = sqrt(2.0);
        double  dn = ZGDN;

        #pragma  omp  parallel  num_threads(NUMTHS)
        {
                //#pragma omp for
                for (int n=0; n<ZGDIRES; n++)
                {
                        gabors[n] = new ZeGabor((CV_PI/ZGDIRES)*n, dn, Sigma, F);
                }

                #pragma omp for
                for (int n=0; n<ZGDIRES; n++)
                {
                        gabors[n]->conv_mat(mat_gray, mat_mags[n]);
                }

                #pragma omp for
                for (int y=0; y<img_gray->height; y++)
                {
                        for (int x=0; x<img_gray->width; x++)
                        {
                                float  sum;
                                float  average;
                                float  max_ve;
                                int    dire;

                                sum = 0.0;
                                dire = 0;

                                max_ve = CV_MAT_ELEM(*mat_mags[0], float, y, x);

                                ZGFeature  *p_features = local_features + y*img_gray->width + x;

                                for (int n=0; n<ZGDIRES; n++)
                                {
                                        float  ve = CV_MAT_ELEM(*mat_mags[n], float, y, x);

                                        p_features->dire_ves[n] = ve;

                                        sum += ve;

                                        if (ve > max_ve)
                                        {
                                                max_ve = ve;
                                                dire = n;
                                        }
                                }

                                average = sum/ZGDIRES;

                                p_features->ve_average = average;
                                p_features->max_ve = max_ve;

                                p_features->mask_noise = 0;

                                if (p_features->max_ve > SCNOISE) p_features->mask_noise = 255;

                                for (int n=0; n<ZGDIRES; n++)
                                {
                                        p_features->mask_dires[n] = 0;
                                        p_features->mask_mdires[n] = 0;
                                }

                                if (x < 10 || x >= img_gray->width - 10 || y < 10 || y >= img_gray->height - 10) continue;

                                for (int n=0; n<ZGDIRES; n++)
                                {
                                        if (p_features->max_ve > SCNOISE
                                            && p_features->dire_ves[n] > SCIFTH*p_features->ve_average)
                                        {
                                                p_features->mask_dires[n] = 255;
                                        }

                                        int  mxy = SCANSTEP+2;

                                        for (int my=-mxy; my<=mxy; my+=2)
                                        {
                                                for(int mx=-mxy; mx<=mxy; mx+=2)
                                                {
                                                        int xdst = x + mx;
                                                        int ydst = y + my;

                                                        ZGFeature  *p_features01 = local_features + ydst*img_gray->width + xdst;

                                                        if (p_features01->max_ve > SCNOISE
                                                            && p_features01->dire_ves[n] > SCIFTH*p_features01->ve_average)
                                                        {
                                                                p_features->mask_mdires[n] = 255;
                                                                goto LOOPILF;
                                                        }
                                                }
                                        }
LOOPILF:                              ;
                                }
                        }
                }
        }


        cvReleaseMat(&mat_gray);
        cvReleaseMat(&mat_edge);
        cvReleaseMat(&mat_dire);
        cvReleaseMat(&mat_mask);

        for (int n=0; n<ZGDIRES; n++)
        {
                delete gabors[n];
                cvReleaseMat(&mat_mags[n]);
        }

        cvReleaseImage(&img_gray);
}

void  scan_whirl_zoom(IplImage  *img_templet, IplImage  *img_src, ZGFeature  *local_features, int  nangle, double zoom01, double zoom02,
                      std::vector<ZPoint>  &infilters, std::vector<ZPoint>  &fifilters, std::vector<ScanPoint>  &scan_points, omp_lock_t *p_lock)
{
        //std::cout << "scan_whirl_zoom" << std::endl;

        int innum_zpoints = infilters.size();
        int finum_zpoints = fifilters.size();

        ScanPoint  scan_point;

        int  innum_prether = (float)innum_zpoints*INZGTHRE;
        int  finum_prether = (float)finum_zpoints*FIZGTHRE;

        std::vector<ZPoint>   zinfilters;
        std::vector<ZPoint>   zfifilters;

        omp_set_lock(p_lock);
        std::cout << "scan  ang : " <<  nangle << std::endl;
        omp_unset_lock(p_lock);

        for (double zoom=zoom01; zoom<zoom02; zoom+=0.1)
        {
                //omp_set_lock(p_lock);
                //std::cout << "scan  ang  zoom : " << nangle << "  " << zoom << std::endl;
                //omp_unset_lock(p_lock);

                zinfilters.clear();
                zfifilters.clear();

                for (std::vector<ZPoint>::iterator  i = infilters.begin();  i != infilters.end();  i++)
                {
                        ZPoint       zfilter;

                        zfilter.a = i->a + (double)nangle/(double)ZGDIRES*CV_PI;
                        zfilter.dire = i->dire+nangle;

                        double  dxr = i->r*zoom*cos(zfilter.a);
                        double  dyr = i->r*zoom*sin(zfilter.a);

                        zfilter.xr = dxr + 0.5;
                        zfilter.yr = dyr + 0.5;

                        zinfilters.push_back(zfilter);
                }

                for (std::vector<ZPoint>::iterator  i = fifilters.begin();  i != fifilters.end();  i++)
                {
                        ZPoint       zfilter;

                        zfilter.a = i->a + (double)nangle/(double)ZGDIRES*CV_PI;
                        zfilter.dire = i->dire+nangle;

                        double  dxr = i->r*zoom*cos(zfilter.a);
                        double  dyr = i->r*zoom*sin(zfilter.a);

                        zfilter.xr = dxr + 0.5;
                        zfilter.yr = dyr + 0.5;

                        zfifilters.push_back(zfilter);
                }

                for (int yz=img_templet->height/2*zoom; yz<img_src->height-img_templet->height/2*zoom; yz+=SCANSTEP)
                {
                        for(int xz=img_templet->width/2*zoom; xz<img_src->width-img_templet->width/2*zoom; xz+=SCANSTEP)
                        {
                                int  num_iden;
                                int  num_uncertainty;

                                num_iden = 0;
                                num_uncertainty = innum_zpoints;

                                for (std::vector<ZPoint>::iterator  i = zinfilters.begin();  i != zinfilters.end();  i++)
                                {
                                        num_uncertainty--;

                                        if (innum_prether > num_iden + num_uncertainty ) break;

                                        int xdst = xz + i->xr;
                                        int ydst = yz - i->yr;

                                        if (xdst <= 10 || ydst <= 10 || xdst >= img_src->width-10 || ydst >= img_src->height-10)  continue;

                                        int  num_match;

                                        int dire = i->dire;

                                        if (dire < 0) dire = dire + ZGDIRES;

                                        dire = dire%ZGDIRES;

                                        ZGFeature  *p_features = local_features + ydst*img_src->width + xdst;

                                        if (p_features->mask_mdires[dire] == 0) continue;

                                        for (int n=0; n<=SCANMXYNI; n++)
                                        {
                                                int mx = scan_imxys[n].x;
                                                int my = scan_imxys[n].y;

                                                int x1 = xdst + mx;
                                                int y1 = ydst + my;

                                                p_features = local_features + y1*img_src->width + x1;

                                                if (p_features->mask_dires[dire] == 255)
                                                {
                                                        num_iden++;
                                                        break;
                                                }
                                        }
                                }

                                if (num_iden < innum_prether) continue;

                                num_iden = 0;
                                num_uncertainty = finum_zpoints;

                                for (std::vector<ZPoint>::iterator  i = zfifilters.begin();  i != zfifilters.end();  i++)
                                {
                                        num_uncertainty--;

                                        if (finum_prether > num_iden + num_uncertainty ) break;

                                        int xdst = xz + i->xr;
                                        int ydst = yz - i->yr;

                                        if (xdst <= 10 || ydst <=  10 || xdst >= img_src->width-10 || ydst >= img_src->height-10)  continue;

                                        int  num_match;

                                        int dire = i->dire;

                                        if (dire < 0) dire = dire + ZGDIRES;

                                        dire = dire%ZGDIRES;

                                        ZGFeature  *p_features = local_features + ydst*img_src->width + xdst;

                                        if (p_features->mask_mdires[dire] == 0) continue;

                                        num_match = 0;

                                        for (int n=0; n<=SCANMXYNF; n++)
                                        {
                                                int mx = scan_fmxys[n].x;
                                                int my = scan_fmxys[n].y;

                                                int x1 = xdst + mx;
                                                int y1 = ydst + my;

                                                p_features = local_features + y1*img_src->width + x1;

                                                if (p_features->mask_dires[dire] == 0)
                                                {
                                                        continue;
                                                }

                                                num_match = 1;

                                                x1 = xdst + mx + scan_dires[dire].x1;
                                                y1 = ydst + my + scan_dires[dire].y1;

                                                p_features = local_features + y1*img_src->width + x1;

                                                if (p_features->mask_dires[dire] == 255)
                                                {
                                                        num_match++;
                                                }

                                                if(num_match >= 2) break;

                                                x1 = xdst + mx + scan_dires[dire].x2;
                                                y1 = ydst + my + scan_dires[dire].y2;

                                                p_features = local_features + y1*img_src->width + x1;

                                                if (p_features->mask_dires[dire] == 255)
                                                {
                                                        num_match++;
                                                }

                                                if(num_match >= 2) break;
                                        }

                                        if (num_match >= 2)
                                        {
                                                num_iden++;
                                        }
                                }

                                if (num_iden >= finum_prether)
                                {
                                        scan_point.x = xz;
                                        scan_point.y = yz;

                                        scan_point.zoom = zoom;
                                        scan_point.na = nangle;
                                        scan_point.va = (double)nangle/(double)ZGDIRES*CV_PI;

                                        scan_point.similarity = (float)num_iden/(float)finum_zpoints;

                                        omp_set_lock(p_lock);

                                        scan_points.push_back(scan_point);

                                        omp_unset_lock(p_lock);
                                }

                        }
                }
        }
}

void  show_zegabor()
{
        ZeGabor   *gabors[ZGDIRES];

        double  Sigma = 2*CV_PI;
        double  F = sqrt(2.0);
        double  dn = ZGDN;

        for (int n=0; n<ZGDIRES; n++)
        {
                gabors[n] = new ZeGabor((CV_PI/ZGDIRES)*n, dn, Sigma, F);
        }

        //CvMat  *mat_kernel = gabors[8]->get_kreal();
        CvMat  *mat_kernel = gabors[ZGDIRES/4]->get_kimag();

        float  illum = 0.0;

        for (int y=0; y<mat_kernel->height; y++)
        {
                for(int x=0; x<mat_kernel->width; x++)
                {
                        float  ve = CV_MAT_ELEM(*mat_kernel, float, y, x);

                        illum += ve;
                }
        }

        std::cout << "illum = " << illum << std::endl;

        CvMat  *mat_show = cvCreateMat(mat_kernel->height, mat_kernel->width, CV_32FC1);

        IplImage   *img_show = cvCreateImage(cvSize(mat_kernel->width, mat_kernel->height), 8, 1);

        cvCopy(mat_kernel, mat_show);

        cvNormalize((CvMat*)mat_show, (CvMat*)mat_show, 0, 255, CV_MINMAX);

        cvConvert(mat_show, img_show);

        cvNamedWindow("show zegabor", 1);
        cvShowImage("show zegabor", img_show);
}

void  debug()
{
        show_zegabor();

        //const char* fn_templet = "./imgs/circ100.jpg";
        //const char* fn_templet = "./imgs/cat3d100.jpg";
        //const char* fn_templet = "./imgs/cof100a.jpg";
        //const char* fn_templet = "./imgs/cat3d150.jpg";
        const char* fn_templet = "./imgs/cof100mw.jpg";
        //const char* fn_templet = "./imgs/beetle200a.jpg";
        //const char* fn_templet = "./imgs/mantis150.jpg";
        //const char* fn_templet = "./imgs/3dfx150.jpg";

        //const char* fn_img = "./imgs/circ140.jpg";
        //const char* fn_img = "./imgs/cat3d100.jpg";
        //const char* fn_img = "./imgs/cof100a.jpg";
        //const char* fn_img = "./imgs/cat3d140.jpg";
        //const char* fn_img = "./imgs/cat3d240.jpg";
        //const char* fn_img = "./imgs/cat100c.jpg";
        const char* fn_img = "./imgs/cof140mw.jpg";
        //const char* fn_img = "./imgs/mantis150.jpg";
        //const char* fn_img = "./imgs/3dfx150.jpg";
        //const char* fn_img = "./imgs/beetle300.jpg";
        //const char* fn_img = "./imgs/test301.jpg";
        //const char* fn_img = "./imgs/scan305.jpg";

        IplImage   *img_templet = cvLoadImage(fn_templet, CV_LOAD_IMAGE_COLOR );

        std::vector<ZPoint>   zfilters;
        std::vector<ZPoint>   zfilters01;
        std::vector<ZPoint>   zfilterweis01;
        std::vector<ZPoint>   zfilterweis02;

        std::vector<ZPoint>   ozfilters;

        std::cout << "train_edge ... " << std::endl;

        IplImage   *img_templet_gray = cvCreateImage( cvGetSize(img_templet), 8, 1 );
        cvCvtColor(img_templet, img_templet_gray, CV_BGR2GRAY);

        //mask_train_edge(img_templet_gray, zfilters);
        train_zfilters(img_templet_gray, zfilters);

        //zfilter_revision(zfilters, zfilterweis, INMAXNZP);
        //zfilter_revision(fizfilters, fizfilterweis, FIMAXNZP);

        IplImage   *img_src = cvLoadImage(fn_img, CV_LOAD_IMAGE_COLOR );
        IplImage   *img_show = cvCloneImage(img_src);

        CvMat       *mat_mask = cvCreateMat(img_src->height, img_src->width, CV_32FC1);

        ZGFeature  *templet_features = new ZGFeature[img_templet->height*img_templet->width];
        ZGFeature  *local_features = new ZGFeature[img_src->height*img_src->width];

        std::cout << "train_zfilters ... " << std::endl;

        train_zfilters(img_templet_gray, zfilters);

        image_local_features(img_templet, templet_features);
        image_local_features(img_src, local_features);

        image_local_features(img_src, local_features);

        std::cout << "classification ... " << std::endl;

        //classification_whirl_zoom(img_templet, img_src, local_features, 0, 1.0, zfilters, out_filter, 4.0);
        classification_whirl_zoom(img_templet, img_templet, templet_features, 0, 1.0, zfilters, zfilters01, 4.0, 3.0, 2);
        //classification_whirl_zoom(img_templet, img_src, local_features, 0, 1.0, zfilterweis, out_filter, INSCIFTH);

        //double similarity = (double)out_filter.size()/(double)zfilterweis.size();

        zfilter_revision(zfilters01, zfilterweis01, INMAXNZP);
        zfilter_revision(zfilters01, zfilterweis02, FIMAXNZP);

        std::vector<ZPoint>   &show_zfilters = zfilterweis02;

/*
void  classification_move_whirl_zoom(IplImage  *img_templet, IplImage  *img_src, ZGFeature  *local_features,
                                int mx, int my, int  nangle, double  zoom, std::vector<ZPoint>  &ifilters,  std::vector<ZPoint>  &ofilter,
                                double  difth01, double  difth02, int  nt_math, double  noise, int  scanmxy)
*/

        classification_move_whirl_zoom(img_templet, img_src, local_features,
                                       0, 0, -1, 1.4, show_zfilters, ozfilters, 3.0, 3.0, 2, SCNOISE , 8);

        for (std::vector<ZPoint>::iterator i = show_zfilters.begin();  i != show_zfilters.end();  i++)
        {
                double  angle = (double)(i->dire)/(double)ZGDIRES*180.0;

                cvEllipse(img_templet, cvPoint(i->x,i->y), cvSize(15,2), -angle, 0.0, 360.0, CV_RGB(255,255,0), -1);

                //cvCircle(img_templet, cvPoint(i->x,i->y), 2, CV_RGB(255,0,0), 2);
        }

        for (std::vector<ZPoint>::iterator i = show_zfilters.begin();  i != show_zfilters.end();  i++)
        {
                double  angle = (double)(i->dire)/(double)ZGDIRES*180.0;

                //cvEllipse(img_templet, cvPoint(i->x,i->y), cvSize(15,2), angle, 0.0, 360.0, CV_RGB(255,255,0), -1);

                cvCircle(img_templet, cvPoint(i->x,i->y), 2, CV_RGB(255,0,0), 2);
        }

        for (std::vector<ZPoint>::iterator i = ozfilters.begin();  i != ozfilters.end();  i++)
        {
                double  angle = (double)(i->dire)/(double)ZGDIRES*180.0;

                cvEllipse(img_show, cvPoint(i->x,i->y), cvSize(15,2), -angle, 0.0, 360.0, CV_RGB(255,255,0), -1);

                //cvCircle(img_show, cvPoint(i->x,i->y), 2, CV_RGB(255,0,0), 2);
        }

        for (std::vector<ZPoint>::iterator i = ozfilters.begin();  i != ozfilters.end();  i++)
        {
                double  angle = (double)(i->dire)/(double)ZGDIRES*180.0;

                //cvEllipse(img_show, cvPoint(i->x,i->y), cvSize(15,2), angle, 0.0, 360.0, CV_RGB(255,255,0), -1);

                cvCircle(img_show, cvPoint(i->x,i->y), 2, CV_RGB(255,0,0), 2);
        }


        std::cout << "zfilters.size = " << zfilters.size() << std::endl;
        std::cout << "zfilters01.size = " << zfilters01.size() << std::endl;
        std::cout << "zfilterweis01.size = " << zfilterweis01.size() << std::endl;
        std::cout << "zfilterweis02.size = " << zfilterweis02.size() << std::endl;
        std::cout << "ozfilters.size = " << ozfilters.size() << std::endl;

        std::cout << "similarity = " << (float)ozfilters.size()/(float)show_zfilters.size() << std::endl;

        cvNamedWindow("templet", 1);
        cvShowImage("templet", img_templet);

        cvNamedWindow("show", 1);
        cvShowImage("show", img_show);

        cvWaitKey(-1);
}

int main(int argc, char ** argv)
{
    if (argc < 3)
    {
                std::cout << "gledcv <template image file> <scan image file>" << std::endl;
                std::cout << "example: gledcv ..\\..\\imgs\\cat3d100.jpg  ..\\..\\imgs\\scan301.jpg" << std::endl;
                return 0;
    }

        //time_t   timer01 = time(0);

        clock_t ck0, ck1, ck_scan, ck_sum;

        ck0 = clock();

        for (int n = 0; n < ZGDIRES; n++)
        {
                double  r = SCANSTEP;

                int     x,y;

                double  dx = cos(CV_PI/32 * n)*r;
                double  dy = sin(CV_PI/32 * n)*r;

                if (dx > 0.0) x = (int)(dx+0.5); else x = (int)(dx-0.5);
                if (dy > 0.0) y = (int)(dy+0.5); else y = (int)(dy-0.5);

                scan_dires[n].x1 = x;
                scan_dires[n].y1 = y;

                scan_dires[n].x2 = -x;
                scan_dires[n].y2 = -y;
        }

        scan_imxys[0].x = 0;
        scan_imxys[0].y = 0;

        for (int n = 1; n <= SCANMXYNI; n++)
        {
                int     x,y;

                double  r = SCANSTEP;

                double  dx = cos(2.0*CV_PI/SCANMXYNI*n)*r;
                double  dy = sin(2.0*CV_PI/SCANMXYNI*n)*r;

                if (dx > 0.0) x = (int)(dx+0.5); else x = (int)(dx-0.5);
                if (dy > 0.0) y = (int)(dy+0.5); else y = (int)(dy-0.5);

                scan_imxys[n].x = x;
                scan_imxys[n].y = y;
        }

        scan_fmxys[0].x = 0;
        scan_fmxys[0].y = 0;

        for (int n = 1; n <= SCANMXYNF; n++)
        {
                int     x,y;

                double  r = SCANSTEP;

                double  dx = cos(2.0*CV_PI/SCANMXYNF*n)*r;
                double  dy = sin(2.0*CV_PI/SCANMXYNF*n)*r;

                if (dx > 0.0) x = (int)(dx+0.5); else x = (int)(dx-0.5);
                if (dy > 0.0) y = (int)(dy+0.5); else y = (int)(dy-0.5);

                scan_fmxys[n].x = x;
                scan_fmxys[n].y = y;
        }

        const char* fn_templet = argv[1];
        const char* fn_img = argv[2];

        IplImage   *img_templet = cvLoadImage(fn_templet, CV_LOAD_IMAGE_COLOR );

        IplImage   *img_templet_gray = cvCreateImage( cvGetSize(img_templet), 8, 1 );
        cvCvtColor(img_templet, img_templet_gray, CV_BGR2GRAY);

        IplImage   *img_src = cvLoadImage(fn_img, CV_LOAD_IMAGE_COLOR );
        IplImage   *img_show = cvCloneImage(img_src);

        std::vector<ZPoint>   zfilters;
        std::vector<ZPoint>   zfilters01;
        std::vector<ZPoint>   zfilterweis01;
        std::vector<ZPoint>   zfilterweis02;

        ZGFeature  *templet_features = new ZGFeature[img_templet->height*img_templet->width];
        ZGFeature  *local_features = new ZGFeature[img_src->height*img_src->width];

        std::cout << "train_zfilters ... " << std::endl;

        train_zfilters(img_templet_gray, zfilters);

        image_local_features(img_templet, templet_features);

        classification_whirl_zoom(img_templet, img_templet, templet_features, 0, 1.0, zfilters, zfilters01, 4.0, 3.0, 2);

        zfilter_revision(zfilters01, zfilterweis01, INMAXNZP);
        zfilter_revision(zfilters01, zfilterweis02, FIMAXNZP);

        std::cout << "zfilters.size = " << zfilters.size() << std::endl;
        std::cout << "zfilters01.size = " << zfilters01.size() << std::endl;
        std::cout << "zfilterweis01.size = " << zfilterweis01.size() << std::endl;
        std::cout << "zfilterweis02.size = " << zfilterweis02.size() << std::endl;

        std::cout << "scan_ ... " << std::endl;

        //time_t   scan_timer = time(0);

        ck1 = clock();
        double   init_timer = double(ck1-ck0)/CLOCKS_PER_SEC;
        ck0 = ck1;

        image_local_features(img_src, local_features);

        ck1 = clock();
        double   itf_timer = double(ck1-ck0)/CLOCKS_PER_SEC;
        ck0 = ck1;

        std::vector<ScanPoint>  fiscan_points;

        omp_lock_t    omplock;

        omp_init_lock(&omplock);

        std::vector<ScanPoint>  scan_points;

        #pragma omp parallel for

        for (int nang=-31; nang<=32; nang++)
        //for (int nang=-10; nang<=10; nang++)
        {
                scan_whirl_zoom(img_templet, img_src, local_features, nang, 0.9, 1.41, zfilterweis01, zfilterweis02, scan_points, &omplock);
        }

        omp_destroy_lock(&omplock);

        //scan_timer = time(0) - scan_timer;

        ck1 = clock();
        double   scan_timer = double(ck1-ck0)/CLOCKS_PER_SEC;
        ck0 = ck1;

        double   timer = init_timer + itf_timer + scan_timer;

        for (std::vector<ScanPoint>::iterator i = scan_points.begin();  i != scan_points.end();  i++)
        {
                //cvCircle(img_show, cvPoint(i->x,i->y), 2, CV_RGB(255,0,0), 2);
                cvCircle(img_show, cvPoint(i->x,i->y), 2, CV_RGB(255,255,0), 2);
                cvCircle(img_show, cvPoint(i->x,i->y), 50, CV_RGB(0,0,255), 2);

                std::cout << "T: " << i->va/CV_PI*180.0 << " "<< i->zoom << " " << i->similarity << " " << i->x << " " << i->y << std::endl;
        }

        cvNamedWindow("img_templet", 1);
        cvShowImage("img_templet", img_templet);

        cvNamedWindow("img_src", 1);
        cvShowImage("img_src", img_src);

        cvNamedWindow("img_show", 1);
        cvShowImage("img_show", img_show);

        std::cout << "init_ " << init_timer << " seconds" << std::endl;
        std::cout << "itf_ " << itf_timer << " seconds" << std::endl;
        std::cout << "scan_ " << scan_timer << " seconds" << std::endl;
        std::cout << "timer : " << timer << " seconds"<< std::endl;
        //std::cout << "timer01 : " << time(0) - timer01 << std::endl;

        cvWaitKey(-1);

        return 0;
}

