#ifndef CKEYFRAME_H
#define CKEYFRAME_H

#include "CLandmark.h"
#include "TypesCloud.h"
#include "../utility/CLogger.h"
#include "CPDescriptorBRIEF.h"



//TODO templatify KeyFrame for maximum generics
#define MAXIMUM_DISTANCE_HAMMING 25
#define MAXIMUM_DISTANCE_HAMMING_PROBABILITY 50
#define BTREE_MAXIMUM_DEPTH 256


#include "CBTree.h"
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h" // defines macros CVXX
#include "DVision/DVision.h"



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

    const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPoolBTree;
    const std::vector< boost::dynamic_bitset< > > vecDescriptorPoolBoW;
    DBoW2::BowVector vecDescriptorPoolB;
    DBoW2::FeatureVector vecDescriptorPoolF;

    const uint32_t uCountInstability;
    const double dMotionScaling;
    std::vector< const CMatchICP* > vecLoopClosures;
    const std::shared_ptr< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > > m_pBTree;

private:

    //ds cloud matching
    static constexpr double m_dCloudMatchingWeightEuclidian        = 10.0;
    static constexpr double m_dCloudMatchingMatchingDistanceCutoff = 75.0;

    std::random_device m_cRandomDevice;

public:

    void saveCloudToFile( ) const;

    //ds offline loading
    std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > getCloudFromFile( const std::string& p_strFile );

    //ds data structure size
    const uint64_t getSizeBytes( ) const;

    const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > getDescriptorPoolBTree( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud );
    const std::vector< boost::dynamic_bitset< > > getDescriptorPoolBoW( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud );


    const std::bitset< DESCRIPTOR_SIZE_BITS > getBitMaskFiltered( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud );
    std::vector< uint32_t > getSplitOrder( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud );
    const std::vector< Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > > getBitProbabilities( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud ) const;
    const std::map< UIDLandmark, CBitStatistics > getBitStatistics( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud ) const;

};

#endif //CKEYFRAME_H
