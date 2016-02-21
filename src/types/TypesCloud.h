#ifndef TYPESCLOUD_H
#define TYPESCLOUD_H

#include "Types.h"

/*struct CDescriptorPoint3DWORLD
{
    const UIDLandmark uID;
    const CPoint3DWORLD vecPointXYZWORLD;
    const CDescriptor matDescriptor;

    CDescriptorPoint3DWORLD( const UIDLandmark& p_uID,
                             const CPoint3DWORLD& p_vecPointXYZWORLD,
                             const CDescriptor& p_matDescriptor ): uID( p_uID ), vecPointXYZWORLD( p_vecPointXYZWORLD ), matDescriptor( p_matDescriptor )
    {
        //ds nothing to do
    }
};*/

struct CDescriptorVectorPoint3DWORLD
{
    const UIDLandmark uID;
    const CPoint3DWORLD vecPointXYZWORLD;
    const CPoint3DCAMERA vecPointXYZCAMERA;
    const cv::Point2d ptUVLEFT;
    const cv::Point2d ptUVRIGHT;
    const std::vector< CDescriptor > vecDescriptors;
    const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecPDescriptorBRIEF;

    CDescriptorVectorPoint3DWORLD( const UIDLandmark& p_uID,
                             const CPoint3DWORLD& p_vecPointXYZWORLD,
                             const CPoint3DCAMERA& p_vecPointXYZCAMERA,
                             const cv::Point2d& p_ptUVLEFT,
                             const cv::Point2d& p_ptUVRIGHT,
                             const std::vector< CDescriptor >& p_vecDescriptors,
                             const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >& p_vecPDescriptorBRIEF ): uID( p_uID ),
                                                                                                              vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                                                                              vecPointXYZCAMERA( p_vecPointXYZCAMERA ),
                                                                                                              ptUVLEFT( p_ptUVLEFT ),
                                                                                                              ptUVRIGHT( p_ptUVRIGHT ),
                                                                                                              vecDescriptors( p_vecDescriptors ),
                                                                                                              vecPDescriptorBRIEF( p_vecPDescriptorBRIEF )
    {
        //ds nothing to do
    }
};

/*struct CDescriptorVectorPointCloud
{
    const UIDCloud uID;
    const Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const std::vector< CDescriptorVectorPoint3DWORLD > vecPoints;

    CDescriptorVectorPointCloud( const UIDCloud& p_uID, const Eigen::Isometry3d& p_matPose, const std::vector< CDescriptorVectorPoint3DWORLD >& p_vecPoints ): uID( p_uID ), matTransformationLEFTtoWORLD( p_matPose ), vecPoints( p_vecPoints )
    {
        //ds nothing to do
    }
};*/

struct CMatchCloud
{
    const CDescriptorVectorPoint3DWORLD* pPointQuery;
    const CDescriptorVectorPoint3DWORLD* pPointReference;
    const double dMatchingDistance;

    CMatchCloud( const CDescriptorVectorPoint3DWORLD* p_pPointQuery,
                 const CDescriptorVectorPoint3DWORLD* p_pPointReference,
                 const double& p_dMatchingDistance ): pPointQuery( p_pPointQuery ), pPointReference( p_pPointReference ), dMatchingDistance( p_dMatchingDistance )
    {
        //ds nothing to do
    }
};

#endif //TYPESCLOUD_H
