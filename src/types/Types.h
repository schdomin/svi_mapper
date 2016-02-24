#ifndef TYPES_H
#define TYPES_H

#include "Typedefs.h"

#define DESCRIPTOR_SIZE_BITS 256
//#define DESCRIPTOR_SIZE_BITS 512
#define DESCRIPTOR_SIZE_BYTES DESCRIPTOR_SIZE_BITS/8



struct CMeasurementLandmark
{
    const UIDLandmark uID;
    const cv::Point2f ptUVLEFT;
    const cv::Point2f ptUVRIGHT;
    const float fDisparity;
    const CPoint3DCAMERA vecPointXYZLEFT;
    const CPoint3DWORLD  vecPointXYZWORLD;
    const CPoint3DWORLD  vecPointXYZWORLDOptimized;
    const Eigen::Isometry3d matTransformationWORLDtoLEFT;
    const MatrixProjection matProjectionWORLDtoLEFT;
    const MatrixProjection matProjectionWORLDtoRIGHT;
    const uint32_t uOptimizations;

    CMeasurementLandmark( const UIDLandmark& p_uID,
                          const cv::Point2f& p_ptUVLEFT,
                          const cv::Point2f& p_ptUVRIGHT,
                          const CPoint3DCAMERA& p_vecPointXYZ,
                          const CPoint3DWORLD& p_vecPointXYZWORLD,
                          const CPoint3DWORLD& p_vecPointXYZWORLDOptimized,
                          const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                          const MatrixProjection& p_matProjectionWORLDtoLEFT,
                          const MatrixProjection& p_matProjectionWORLDtoRIGHT,
                          const uint32_t& p_uOptimizations ): uID( p_uID ),
                                                              ptUVLEFT( p_ptUVLEFT ),
                                                              ptUVRIGHT( p_ptUVRIGHT ),
                                                              fDisparity( p_ptUVLEFT.x-p_ptUVRIGHT.x ),
                                                              vecPointXYZLEFT( p_vecPointXYZ ),
                                                              vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                              vecPointXYZWORLDOptimized( p_vecPointXYZWORLDOptimized ),
                                                              matTransformationWORLDtoLEFT( p_matTransformationWORLDtoLEFT ),
                                                              matProjectionWORLDtoLEFT( p_matProjectionWORLDtoLEFT ),
                                                              matProjectionWORLDtoRIGHT( p_matProjectionWORLDtoRIGHT ),
                                                              uOptimizations( p_uOptimizations )
    {
        //ds input validation
        assert( ptUVLEFT.y == ptUVRIGHT.y );
        assert( ptUVLEFT.x > ptUVRIGHT.x );
        assert( 0.0 < vecPointXYZLEFT.z( ) );
        assert( 0.0f < fDisparity );
    }

};

struct CMatchTracking
{
    const cv::KeyPoint cKeyPoint;
    const CDescriptor matDescriptor;

    CMatchTracking( const cv::KeyPoint& p_cKeyPoint, const CDescriptor& p_matDescriptor ): cKeyPoint( p_cKeyPoint ), matDescriptor( p_matDescriptor )
    {
        //ds nothing to do
    }
};

struct CMatchTriangulation
{
    const CPoint3DCAMERA vecPointXYZCAMERA;
    const cv::Point2f ptUVCAMERA;
    const CDescriptor matDescriptorCAMERA;

    CMatchTriangulation( const CPoint3DCAMERA& p_vecPointXYZCAMERA,
                         const cv::Point2f& p_ptUVCAMERA,
                         const CDescriptor& p_matDescriptorCAMERA ): vecPointXYZCAMERA( p_vecPointXYZCAMERA ),
                                                                     ptUVCAMERA( p_ptUVCAMERA ),
                                                                     matDescriptorCAMERA( p_matDescriptorCAMERA )
    {
        //ds no input validation here (negative z allowed)
    }
};

struct CBitStatistics
{
    Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecBitProbabilities;
    Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecBitPermanences;

    CBitStatistics( const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >& p_vecBitProbabilities,
                    const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >& p_vecBitPermanences ): vecBitProbabilities( p_vecBitProbabilities ),
                                                                                                   vecBitPermanences( p_vecBitPermanences )
    {

    }
};

#endif //TYPES_H
