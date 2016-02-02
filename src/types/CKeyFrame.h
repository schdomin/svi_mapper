#ifndef CKEYFRAME_H
#define CKEYFRAME_H

#include "CLandmark.h"
#include "TypesCloud.h"
#include "../utility/CLogger.h"



class CKeyFrame
{

public:

    //ds keyframe loop closing
    struct CMatchICP
    {
        const CKeyFrame* pKeyFrameReference;
        const Eigen::Isometry3d matTransformationToClosure;
        const std::shared_ptr< const std::vector< CMatchCloud > > vecMatches;

        CMatchICP( const CKeyFrame* p_pKeyFrameReference,
                   const Eigen::Isometry3d& p_matTransformationToReference,
                   const std::shared_ptr< const std::vector< CMatchCloud > > p_vecMatches ): pKeyFrameReference( p_pKeyFrameReference ),
                                                                                             matTransformationToClosure( p_matTransformationToReference ),
                                                                                             vecMatches( p_vecMatches )
        {
            //ds nothing to do
        }
    };

public:

    //ds key frame with closure
    CKeyFrame( const std::vector< CKeyFrame* >::size_type& p_uID,
               const uint64_t& p_uFrame,
               const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
               const CLinearAccelerationIMU& p_vecLinearAcceleration,
               const std::vector< const CMeasurementLandmark* >& p_vecMeasurements,
               const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud,
               const uint32_t& p_uCountInstability,
               const double& p_dMotionScaling,
               const std::vector< const CMatchICP* > p_vecLoopClosures );

    //ds key frame without closure
    CKeyFrame( const std::vector< CKeyFrame* >::size_type& p_uID,
               const uint64_t& p_uFrame,
               const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
               const CLinearAccelerationIMU& p_vecLinearAcceleration,
               const std::vector< const CMeasurementLandmark* >& p_vecMeasurements,
               const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud,
               const uint32_t& p_uCountInstability,
               const double& p_dMotionScaling );

    //ds key frame loading from file (used for offline cloud matching)
    CKeyFrame( const std::string& p_strFile );

    ~CKeyFrame( );

public:

    const std::vector< CKeyFrame* >::size_type uID;
    const uint64_t uFrameOfCreation;
    Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const CLinearAccelerationIMU vecLinearAccelerationNormalized;
    const std::vector< const CMeasurementLandmark* > vecMeasurements;
    bool bIsOptimized = false;
    const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > vecCloud;
    std::map< UIDDescriptor, const CDescriptorVectorPoint3DWORLD* > mapDescriptorToPoint;
    const std::vector< CDescriptorBRIEF > vecDescriptorPool;
    const CDescriptors vecDescriptorPoolCV;
    const uint32_t uCountInstability;
    const double dMotionScaling;
    std::vector< const CMatchICP* > vecLoopClosures;

private:

    //ds cloud matching
    static constexpr double m_dCloudMatchingWeightEuclidian        = 10.0;
    static constexpr double m_dCloudMatchingMatchingDistanceCutoff = 75.0;

public:

    void saveCloudToFile( ) const;
    std::shared_ptr< const std::vector< CMatchCloud > > getMatchesVisualSpatial( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloudQuery ) const;

    //ds offline loading
    std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > getCloudFromFile( const std::string& p_strFile );

    //ds data structure size
    const uint64_t getSizeBytes( ) const;

    //ds full descriptor pool (getDescriptorPool MUST be called before getDescriptorPoolCV to set up the descriptor-to-point map)
    const std::vector< CDescriptorBRIEF > getDescriptorPool( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud );
    const CDescriptors getDescriptorPoolCV( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud );

};

#endif //CKEYFRAME_H
