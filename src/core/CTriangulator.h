#ifndef CTRIANGULATOR_H
#define CTRIANGULATOR_H

#include "vision/CStereoCamera.h"



class CTriangulator
{

//ds ctor/dtor
public:

    CTriangulator( const std::shared_ptr< CStereoCamera > p_pStereoCamera,
                   const std::shared_ptr< cv::DescriptorExtractor > p_pExtractor );
    ~CTriangulator( );

//ds defines
public:

    static constexpr float fMinimumSearchRangePixels = 60.0;
    static constexpr double dMinimumDisparityPixels  = 0.01;

//ds members
private:

    //ds cameras
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;

    //ds matching
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const float m_fMatchingDistanceCutoff;


    //ds intrinsics
    const double m_dF;
    const double m_dFInverse;
    const double m_dPu;
    const double m_dPv;
    const double m_dDuR;
    const double m_dDuRFlipped;

public:

    const double dDepthMinimumMeters;
    const double dDepthMaximumMeters;

//ds api
public:

    //ds triangulation methods
    const CMatchTriangulation getPointTriangulatedInRIGHTFull( cv::Mat& p_matDisplaySTEREO, const cv::Mat& p_matImageRIGHT,
            const float& p_fUTopLeft,
            const float& p_fVTopLeft,
            const float& p_fKeyPointSizePixels,
            const cv::Point2f& p_ptUVLEFT,
            const CDescriptor& p_matReferenceDescriptorLEFT ) const;
    const CMatchTriangulation getPointTriangulatedInLEFT( const cv::Mat& p_matImageLEFT,
            const float& p_fUTopLeft,
            const float& p_fVTopLeft,
            const float& p_fKeyPointSizePixels,
            const cv::Point2f& p_ptUVRIGHT,
            const CDescriptor& p_matReferenceDescriptorRIGHT ) const;

    const CMatchTriangulation getPointTriangulatedInRIGHT( const cv::Mat& p_matImageRIGHT,
            const float& p_fUTopLeft,
            const float& p_fVTopLeft,
            const float& p_fKeyPointSizePixels,
            const cv::Point2f& p_ptUVLEFT,
            const CDescriptor& p_matReferenceDescriptorLEFT ) const;
    const CMatchTriangulation getPointTriangulatedInLEFT( const cv::Mat& p_matImageLEFT,
            const float& p_fSearchRange,
            const float& p_fUTopLeft,
            const float& p_fVTopLeft,
            const float& p_fKeyPointSizePixels,
            const cv::Point2f& p_ptUVRIGHT,
            const CDescriptor& p_matReferenceDescriptorRIGHT ) const;

    const CPoint3DCAMERA getPointInLEFT( const cv::Point2f& p_ptUVLEFT, const cv::Point2f& p_ptUVRIGHT ) const;

private:

    friend class CMatcherEpipolar;
    friend class CFundamentalMatcher;
};

#endif //#define CTRIANGULATOR_H
