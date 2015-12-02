#ifndef CMAPPER_H
#define CMAPPER_H

#include "types/CKeyFrame.h"
#include "types/CTypesThreading.h"



class CMapper
{

//ds ctor/dtor
public:

    CMapper( std::shared_ptr< CHandleLandmarks > p_hLandmarks,
             std::shared_ptr< CHandleKeyFrames > p_hKeyFrames,
             std::shared_ptr< CHandleOptimization > p_hOptimization,
             std::shared_ptr< CHandleMapUpdate > p_hMapUpdate );
    ~CMapper( );

//ds access
public:

    //ds add new (chunks of) key frames
    void addKeyFramesSorted( const std::vector< CKeyFrame* >& p_vecKeyFramesToAdd );

    //ds compute loop closures for new key frames and add them
    void integrateAddedKeyFrames( );

    //ds checks if an optimization is required and requests one if so
    const bool checkAndRequestOptimization( );

    //ds check for map update
    void checkForMapUpdate( );

//ds internals
private:

    //ds general
    std::shared_ptr< CHandleLandmarks > m_hLandmarks;
    std::shared_ptr< CHandleKeyFrames > m_hKeyFrames;
    std::shared_ptr< CHandleOptimization > m_hOptimization;
    const std::shared_ptr< CHandleMapUpdate > m_hMapUpdate;

    //ds transfer buffer of key frames to guarantee minimum critical sections in the thread
    std::vector< CKeyFrame* > m_vecBufferAddedKeyFrames;

    //ds loop closing
    const std::vector< CKeyFrame* >::size_type m_uMinimumLoopClosingKeyFrameDistance = 20; //20
    const std::vector< CMatchCloud >::size_type m_uMinimumNumberOfMatchesLoopClosure = 25; //25
    const std::vector< CKeyFrame* >::size_type m_uLoopClosingKeyFrameWaitingQueue    = 1;
    std::vector< CKeyFrame* >::size_type m_uLoopClosingKeyFramesInQueue              = 0;
    std::vector< CKeyFrame* >::size_type m_uIDLoopClosureOptimizedLAST               = 0;
    const double m_dLoopClosingRadiusSquaredMeters                                   = 1000.0;

    //ds optimization triggering
    const std::vector< CLandmark* >::size_type m_uMinimumLandmarksForKeyFrame    = 50;
    std::vector< CKeyFrame* >::size_type m_uIDProcessedKeyFrameLAST              = 0;
    const std::vector< CKeyFrame* >::size_type m_uIDDeltaKeyFrameForOptimization = 20; //10
    const double m_dMaximumMotionScalingForOptimization = 1.05;
    Eigen::Vector3d m_vecTranslationToG2o;
    bool m_bOptimizationPending = false;

    //ds timing
    double m_dDurationTotalSeconds = 0.0;

//ds helpers
private:

    const std::vector< const CKeyFrame::CMatchICP* > _getLoopClosuresForKeyFrame( const CKeyFrame* p_pKeyFrame,
                                                                                  const double& p_dSearchRadiusMeters,
                                                                                  const std::vector< CMatchCloud >::size_type& p_uMinimumNumberOfMatchesLoopClosure );

};

#endif //CMAPPER_H
