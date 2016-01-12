#include "CMapper.h"
#include "vision/CMiniVisionToolbox.h"



CMapper::CMapper( std::shared_ptr< CHandleLandmarks > p_hLandmarks,
                  std::shared_ptr< CHandleKeyFrames > p_hKeyFrames,
                  std::shared_ptr< CHandleMapping > p_hMapping,
                  std::shared_ptr< CHandleOptimization > p_hOptimization ): m_hLandmarks( p_hLandmarks ),
                                                                      m_hKeyFrames( p_hKeyFrames ),
                                                                      m_hMapping( p_hMapping ),
                                                                      m_hOptimization( p_hOptimization )
{
    m_vecBufferAddedKeyFrames.clear( );

    //ds configuration
    CLogger::openBox( );
    std::printf( "[1]<CMapper>(CMapper) minimum key frame distance: %lu\n", m_uMinimumLoopClosingKeyFrameDistance );
    std::printf( "[1]<CMapper>(CMapper) minimum matches for closure: %lu\n", m_uMinimumNumberOfMatchesLoopClosure );
    std::printf( "[1]<CMapper>(CMapper) closure search radius: %fm\n", m_dLoopClosingRadiusSquaredMeters );
    std::printf( "[1]<CMapper>(CMapper) instance allocated\n" );
    CLogger::closeBox( );
}

CMapper::~CMapper( )
{
    //ds info
    std::printf( "[1]<CMapper>(CMapper) total computation time: %fs\n", m_dDurationTotalSeconds );
    std::printf( "[1]<CMapper>(CMapper) instance deallocated\n" );
}

void CMapper::addKeyFramesSorted( const std::vector< CKeyFrame* >& p_vecKeyFramesToAdd )
{
    //ds copy the vector
    m_vecBufferAddedKeyFrames = p_vecKeyFramesToAdd;
    //std::printf( "[%06lu]<CMapper>(addKeyFramesSorted) added key frames: [%06lu] to [%06lu] (%lu)\n", m_uFrameID, p_vecKeyFramesToAdd.front( )->uID, p_vecKeyFramesToAdd.back( )->uID, p_vecKeyFramesToAdd.size( ) );
}

void CMapper::integrateAddedKeyFrames( )
{
    //ds loop over the buffer
    for( CKeyFrame* pKeyFrame: m_vecBufferAddedKeyFrames )
    {
        //ds compute loop closures
        pKeyFrame->vecLoopClosures = _getLoopClosuresForKeyFrame( pKeyFrame, m_dLoopClosingRadiusSquaredMeters, m_uMinimumNumberOfMatchesLoopClosure );

        //ds info
        if( 0 < pKeyFrame->vecLoopClosures.size( ) )
        {
            std::printf( "[1]<CMapper>(integrateAddedKeyFrames) key frame [%06lu] loop closures: %lu\n", pKeyFrame->uID, pKeyFrame->vecLoopClosures.size( ) );

            //ds relevant for optimization
            ++m_uLoopClosingKeyFramesInQueue;
        }

        //ds add final key frame
        m_hKeyFrames->vecKeyFrames->push_back( pKeyFrame );
    }
}

const bool CMapper::checkAndRequestOptimization( )
{
    //ds must be filled
    if( 1 < m_hKeyFrames->vecKeyFrames->size( ) && !m_bOptimizationPending )
    {
        //ds extract information from the keyframes (overview)
        const double dMotionScalingCurrent  = m_hKeyFrames->vecKeyFrames->back( )->dMotionScaling;
        const double dMotionScalingPrevious = ( m_hKeyFrames->vecKeyFrames->back( )-1 )->dMotionScaling;

        //ds check if we are not in a critical situation before triggering an optimization
        if( m_dMaximumMotionScalingForOptimization > ( dMotionScalingCurrent+dMotionScalingPrevious )/2.0 && 0 == m_hKeyFrames->vecKeyFrames->back( )->uCountInstability )
        {
            //ds current key frame id
            const std::vector< CLandmark* >::size_type uIDKeyFrameCurrent = m_hKeyFrames->vecKeyFrames->back( )->uID;

            //ds check if optimization is required (based on key frame id or loop closing) TODO beautify this case
            if( m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDProcessedKeyFrameLAST                                                                              ||
               ( m_uLoopClosingKeyFrameWaitingQueue < m_uLoopClosingKeyFramesInQueue && m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDLoopClosureOptimizedLAST ) )
            {
                //ds we will trigger no new optimizations until this one is completed
                m_bOptimizationPending = true;

                //ds compute landmark begin ID
                const std::vector< CLandmark* >::size_type uIDBeginLandmark = std::min( m_hKeyFrames->vecKeyFrames->at( m_uIDProcessedKeyFrameLAST )->vecMeasurements.front( )->uID, m_hLandmarks->vecLandmarks->back( )->uID );

                {
                    //ds request optimization - lock
                    std::lock_guard< std::mutex > cLockGuardOptimization( m_hOptimization->cMutex );

                    //ds set request information
                    m_hOptimization->cRequest.uFrame                = m_hKeyFrames->vecKeyFrames->back( )->uFrameOfCreation;
                    m_hOptimization->cRequest.uIDBeginKeyFrame      = m_uIDProcessedKeyFrameLAST;
                    m_hOptimization->cRequest.uIDBeginLandmark      = uIDBeginLandmark;
                    m_hOptimization->cRequest.vecTranslationToG2o   = Eigen::Vector3d::Zero( );
                    m_hOptimization->cRequest.uLoopClosureKeyFrames = m_uLoopClosingKeyFramesInQueue;

                    std::printf( "[1][%06lu]<CMapper>(checkAndRequestOptimization) requesting optimization for key frames [%06lu] to [%06lu]\n",
                                  m_hOptimization->cRequest.uFrame, m_hOptimization->cRequest.uIDBeginKeyFrame, uIDKeyFrameCurrent );

                    //ds enable processing
                    m_hOptimization->bRequestProcessed = false;
                }

                //ds notify optimizer
                m_hOptimization->cConditionVariable.notify_one( );

                //ds if loop closure triggered
                if( 0 < m_uLoopClosingKeyFramesInQueue )
                {
                    m_uIDLoopClosureOptimizedLAST = uIDKeyFrameCurrent;
                }

                //ds update counters
                m_uLoopClosingKeyFramesInQueue = 0;

                return true;
            }
        }
    }

    return false;
}

void CMapper::updateMap( )
{
    std::printf( "[1][%06lu]<CMapper>(updateMap) received map update - key frame delta: %lu\n",
                 m_hMapping->cMapUpdate.uIDFrame, m_hKeyFrames->vecKeyFrames->back( )->uID-m_hMapping->cMapUpdate.uIDKeyFrame );

    //ds update optimization criteria
    m_uIDProcessedKeyFrameLAST= m_hMapping->cMapUpdate.uIDKeyFrame+1;

    //ds done
    m_bOptimizationPending = false;
}

//ds locked key frames from upper scope
const std::vector< const CKeyFrame::CMatchICP* > CMapper::_getLoopClosuresForKeyFrame( const CKeyFrame* p_pKeyFrame,
                                                                                       const double& p_dSearchRadiusMeters,
                                                                                       const std::vector< CMatchCloud >::size_type& p_uMinimumNumberOfMatchesLoopClosure )
{
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds count attempts
    //uint32_t uOptimizationAttempts = 0;

    //ds buffer constants (READ ONLY)
    const std::vector< CKeyFrame* >::size_type uIDQuery       = p_pKeyFrame->uID;
    const Eigen::Isometry3d matTransformationLEFTtoWORLDQuery = p_pKeyFrame->matTransformationLEFTtoWORLD;
    const Eigen::Vector3d vecPositionQuery                    = matTransformationLEFTtoWORLDQuery.translation( );
    const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > vecPointsQuery = p_pKeyFrame->vecCloud;

    //ds potential keyframes for loop closing
    std::vector< const CKeyFrame* > vecPotentialClosureKeyFrames;

    //ds check all keyframes for distance
    for( const CKeyFrame* pKeyFrameReference: *m_hKeyFrames->vecKeyFrames )
    {
        //ds break if near to current id (forward closing)
        if( m_uMinimumLoopClosingKeyFrameDistance > uIDQuery-pKeyFrameReference->uID )
        {
            break;
        }

        //ds if the distance is acceptable
        if( p_dSearchRadiusMeters > ( pKeyFrameReference->matTransformationLEFTtoWORLD.translation( )-vecPositionQuery ).squaredNorm( ) )
        {
            //ds add the keyframe to the loop closing pool
            vecPotentialClosureKeyFrames.push_back( pKeyFrameReference );
        }
    }

    //ds solution vector
    std::vector< const CKeyFrame::CMatchICP* > vecLoopClosures( 0 );

    //std::printf( "<CTrackerStereo>(_getLoopClosureKeyFrameFCFS) checking for closure in potential keyframes: %lu\n", vecPotentialClosureKeyFrames.size( ) );

    //ds compare current cloud against previous ones to enable loop closure (skipping the keyframes added just before)
    for( const CKeyFrame* pKeyFrameReference: vecPotentialClosureKeyFrames )
    {
        //ds get matches
        std::shared_ptr< const std::vector< CMatchCloud > > vecMatches( pKeyFrameReference->getMatches( vecPointsQuery ) );

        //ds if we have a suffient amount of matches
        if( p_uMinimumNumberOfMatchesLoopClosure < vecMatches->size( ) )
        {
            //std::printf( "<CTrackerStereo>(_getLoopClosureKeyFrameFCFS) found closure: [%06lu] > [%06lu] matches: %lu\n", p_uID, pKeyFrameReference->uID, vecMatches->size( ) );

            //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Isometry3d matTransformationToClosure( pKeyFrameReference->matTransformationLEFTtoWORLD.inverse( )*matTransformationLEFTtoWORLDQuery );
            matTransformationToClosure.translation( ) = Eigen::Vector3d::Zero( );
            //const Eigen::Isometry3d matTransformationToClosureInitial( matTransformationToClosure );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 0.25; //0.25
            const double dMaximumErrorAverageForClosure = 0.2; //0.1

            //ds LS setup
            Eigen::Matrix< double, 6, 6 > matH;
            Eigen::Matrix< double, 6, 1 > vecB;
            Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );

            //std::printf( "<CTrackerStereo>(_getLoopClosureKeyFrameFCFS) t: %4.1f %4.1f %4.1f > ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uInliers         = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( matTransformationToClosure*cMatch.cPointQuery.vecPointXYZCAMERA );
                    if( 0.0 < vecPointXYZQuery.z( ) )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds adjust omega to inverse depth value (the further away the point, the less weight)
                        matOmega(2,2) = 1.0/( cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error
                        const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.cPointReference.vecPointXYZCAMERA );

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                        }
                        else
                        {
                            ++uInliers;
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                        Eigen::Matrix< double, 3, 6 > matJacobianTransform;
                        matJacobianTransform.setZero( );
                        matJacobianTransform.block<3,3>(0,0).setIdentity( );
                        matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZQuery );

                        //ds precompute transposed
                        const Eigen::Matrix< double, 6, 3 > matJacobianTransformTransposed( matJacobianTransform.transpose( ) );

                        //ds accumulate
                        matH += dWeight*matJacobianTransformTransposed*matOmega*matJacobianTransform;
                        vecB += dWeight*matJacobianTransformTransposed*matOmega*vecError;
                    }
                }

                //ds solve the system and update the estimate
                matTransformationToClosure = CMiniVisionToolbox::getTransformationFromVector( matH.ldlt( ).solve( -vecB ) )*matTransformationToClosure;

                //ds enforce rotation symmetry
                const Eigen::Matrix3d matRotation        = matTransformationToClosure.linear( );
                Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                matRotationSquared.diagonal( ).array( ) -= 1;
                matTransformationToClosure.linear( )    -= 0.5*matRotation*matRotationSquared;

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/vecMatches->size( );

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && p_uMinimumNumberOfMatchesLoopClosure < uInliers )
                    {
                        //std::printf( "<CMapper>(_getLoopClosuresForKeyFrame) found closure: [%06lu] > [%06lu] (matches: %3lu, iterations: %2u, average error: %5.3f, inliers: %2u)\n",
                        //             uIDQuery, pKeyFrameReference->uID, vecMatches->size( ), uLS, dErrorAverage, uInliers );
                        vecLoopClosures.push_back( new CKeyFrame::CMatchICP( pKeyFrameReference, matTransformationToClosure, vecMatches ) );
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        //std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );
                        //std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    //std::printf( "system did not converge\n" );
                }
            }
        }

        //ds escape if we have enough closures
        if( m_uMaximumNumberOfLoopClosuresPerKF == vecLoopClosures.size( ) )
        {
            break;
        }
    }

    //ds info
    m_dDurationTotalSeconds += CTimer::getTimeSeconds( )-dTimeStartSeconds;

    //ds return found closures
    return vecLoopClosures;
}
