#include "Cg2oOptimizer.h"

#include <dirent.h>
#include <stdio.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "exceptions/CExceptionLogfileTree.h"



//ds register custom g2o types
namespace g2o
{
    G2O_REGISTER_TYPE( EDGE_SE3_LINEAR_ACCELERATION, EdgeSE3LinearAcceleration )

    //ds if drawing is enabled
    #ifdef G2O_HAVE_OPENGL

        G2O_REGISTER_ACTION( EdgeSE3LinearAccelerationDrawAction )

    #endif
}



//ds CONTINOUS CONSTRUCTOR
Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames,
                              const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDInitial ): Cg2oOptimizer( p_pCameraSTEREO, p_vecLandmarks, p_vecKeyFrames )
{
    CLogger::openBox( );
    m_uIDKeyFrameFrom = -1;

    //ds trajectory only
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+1 );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );
    m_cOptimizerSparseTrajectoryOnly.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );

    //ds add the first pose separately
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+m_uIDShift );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );

    //ds set to graph
    m_cOptimizerSparse.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );
    m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseFIRSTNOTINGRAPH, Eigen::Vector3d( 0.0, 0.0, 0.0 ) ) );

    //ds get a copy
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+m_uIDShift );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );

    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

//ds TAILWISE CONSTRUCTOR
Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                        m_vecLandmarks( p_vecLandmarks ),
                                                                                        m_vecKeyFrames( p_vecKeyFrames ),
                                                                                        m_matInformationPose( 100000*Eigen::Matrix< double, 6, 6 >::Identity( ) ),
                                                                                        m_matInformationLoopClosure( _getInformationNoZ( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*m_matInformationPose ) ) ),
                                                                                        m_matInformationLandmarkClosure( 1000*Eigen::Matrix< double, 3, 3 >::Identity( ) )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );
    m_vecPoseEdges.clear( );
    m_mapEdgeIDs.clear( );

    //ds configure the solver (var_cholmod)
    //m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType >( ) );
    //g2o::BlockSolver_6_3* pSolver( new g2o::BlockSolver_6_3( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds trajectory only
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolverTrajectoryOnly( new g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType >( ) );
    g2o::BlockSolver_6_3* pSolverTrajectoryOnly( new g2o::BlockSolver_6_3( pLinearSolverTrajectoryOnly ) );
    g2o::OptimizationAlgorithmGaussNewton* pAlgorithmTrajectoryOnly( new g2o::OptimizationAlgorithmGaussNewton( pSolverTrajectoryOnly ) );
    //g2o::OptimizationAlgorithmLevenberg* pAlgorithmTrajectoryOnly( new g2o::OptimizationAlgorithmLevenberg( pSolverTrajectoryOnly ) );
    m_cOptimizerSparseTrajectoryOnly.setAlgorithm( pAlgorithmTrajectoryOnly );

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    m_cOptimizerSparse.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_pCameraLEFT->m_dFyP, m_pCameraSTEREO->m_pCameraLEFT->m_dCxP, m_pCameraSTEREO->m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    m_cOptimizerSparse.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_pCameraSTEREO->m_pCameraRIGHT->m_dFxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dFyP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCyP );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    m_cOptimizerSparse.addParameter( pCameraParametersRIGHT );

    //ds imu offset (as IMU to LEFT) EMPTY FUNCTIONALITY FOR CAMERAS WITHOUT IMU
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( Eigen::Isometry3d( Eigen::Matrix4d::Identity( ) ) );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    m_cOptimizerSparse.addParameter( pOffsetIMUtoLEFT );

    CLogger::openBox( );
    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod\n" );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

//ds CONTINOUS CONSTRUCTOR
Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCameraIMU > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames,
                              const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDInitial ): Cg2oOptimizer( p_pCameraSTEREO, p_vecLandmarks, p_vecKeyFrames )
{
    CLogger::openBox( );
    m_uIDKeyFrameFrom = -1;

    //ds trajectory only
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+1 );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );
    m_cOptimizerSparseTrajectoryOnly.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );

    //ds add the first pose separately
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+m_uIDShift );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );

    //ds set to graph
    m_cOptimizerSparse.addVertex( m_pVertexPoseFIRSTNOTINGRAPH );
    m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( m_pVertexPoseFIRSTNOTINGRAPH, Eigen::Vector3d( 0.0, -1.0, 0.0 ) ) );

    //ds get a copy
    m_pVertexPoseFIRSTNOTINGRAPH = new g2o::VertexSE3( );
    m_pVertexPoseFIRSTNOTINGRAPH->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseFIRSTNOTINGRAPH->setId( m_uIDKeyFrameFrom+m_uIDShift );
    m_pVertexPoseFIRSTNOTINGRAPH->setFixed( true );

    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

//ds TAILWISE CONSTRUCTOR
Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCameraIMU > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                        m_vecLandmarks( p_vecLandmarks ),
                                                                                        m_vecKeyFrames( p_vecKeyFrames ),
                                                                                        m_matInformationPose( 100000*Eigen::Matrix< double, 6, 6 >::Identity( ) ),
                                                                                        m_matInformationLoopClosure( _getInformationNoZ( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*m_matInformationPose ) ) ),
                                                                                        m_matInformationLandmarkClosure( 1000*Eigen::Matrix< double, 3, 3 >::Identity( ) )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );
    m_vecPoseEdges.clear( );
    m_mapEdgeIDs.clear( );

    //ds configure the solver (var_cholmod)
    //m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType >( ) );
    //g2o::BlockSolver_6_3* pSolver( new g2o::BlockSolver_6_3( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds trajectory only
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolverTrajectoryOnly( new g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType >( ) );
    g2o::BlockSolver_6_3* pSolverTrajectoryOnly( new g2o::BlockSolver_6_3( pLinearSolverTrajectoryOnly ) );
    g2o::OptimizationAlgorithmGaussNewton* pAlgorithmTrajectoryOnly( new g2o::OptimizationAlgorithmGaussNewton( pSolverTrajectoryOnly ) );
    //g2o::OptimizationAlgorithmLevenberg* pAlgorithmTrajectoryOnly( new g2o::OptimizationAlgorithmLevenberg( pSolverTrajectoryOnly ) );
    m_cOptimizerSparseTrajectoryOnly.setAlgorithm( pAlgorithmTrajectoryOnly );

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    m_cOptimizerSparse.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_pCameraLEFT->m_dFyP, m_pCameraSTEREO->m_pCameraLEFT->m_dCxP, m_pCameraSTEREO->m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    m_cOptimizerSparse.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_pCameraSTEREO->m_pCameraRIGHT->m_dFxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dFyP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCyP );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    m_cOptimizerSparse.addParameter( pCameraParametersRIGHT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_pCameraSTEREO->m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    m_cOptimizerSparse.addParameter( pOffsetIMUtoLEFT );

    CLogger::openBox( );
    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod\n" );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

Cg2oOptimizer::~Cg2oOptimizer( )
{
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) total computation time: %fs\n", m_dDurationTotalSecondsOptimization );
    std::printf( "[0]<Cg2oOptimizer>(Cg2oOptimizer) instance deallocated\n" );
}

void Cg2oOptimizer::optimize( const UIDFrame& p_uFrameCount,
                              const UIDKeyFrame& p_uIDBeginKeyFrame,
                              const UIDKeyFrame& p_uNumberOfClosedKeyFrames,
                              const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds derive begin landmark within optimized chunk
    const UIDLandmark uIDBeginLandmark = m_vecKeyFrames->at( p_uIDBeginKeyFrame )->vecMeasurements.front( )->uID;

    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    assert( uIDBeginLandmark < m_vecKeyFrames->back( )->vecMeasurements.back( )->uID );

    //ds check if we have a loop closure (landmark free optimization)
    if( 0 < p_uNumberOfClosedKeyFrames )
    {
        //ds closure count
        uint32_t uLoopClosuresTotal = 0;

        //ds loop closure holder
        std::vector< CLoopClosureRaw > vecLoopClosures;

        //ds compute save file name
        char chBufferLC[256];
        std::snprintf( chBufferLC, 256, "g2o/local/keyframes_%06lu-%06lu", m_vecKeyFrames->at( p_uIDBeginKeyFrame )->uID, m_vecKeyFrames->back( )->uID );
        const std::string strSaveFilePrefixLC = chBufferLC;

        //ds loop over the camera vertices vector
        for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < m_vecKeyFrames->size( ); ++uID )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_uIDKeyFrameFrom, pKeyFrame, p_vecTranslationToG2o );

            //ds closed edges
            OptimizableGraph::EdgeSet vecLoopClosureEdges;

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                //ds accumulate the elements
                vecLoopClosureEdges.insert( _getEdgeLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure ) );
                ++uLoopClosuresTotal;

                //ds get loop closure
                //g2o::EdgeSE3* pEdgeLoopClosure = _getEdgeLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure );

                //ds add to graph
                //m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
                //vecLoopClosures.push_back( CLoopClosureRaw( pClosure->pKeyFrameReference->uID, pKeyFrame->uID, pEdgeLoopClosure ) );
            }

            //ds if there were any loop closures for this keyframe
            if( !vecLoopClosureEdges.empty( ) )
            {
                const g2o::HyperGraph::VertexIDMap::iterator itPoseCurrent( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );
                assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseCurrent );

                m_cBufferClosures.addEdgeSet( vecLoopClosureEdges );
                m_cBufferClosures.addVertex( dynamic_cast< g2o::VertexSE3* >( itPoseCurrent->second ) );
            }

            //ds evaluate loop closure window
            if( m_cBufferClosures.checkList( p_uNumberOfClosedKeyFrames ) )
            {
                //ds determine valid closures
                m_cClosureChecker.init( m_cBufferClosures.vertices( ), m_cBufferClosures.edgeSet( ), m_dMaximumThresholdLoopClosing );
                m_cClosureChecker.check( );

                std::printf( "WINDOW SIZE: %lu\n", p_uNumberOfClosedKeyFrames );
                std::printf( "INLIERS: %i\n", m_cClosureChecker.inliers( ) );

                //ds if at least one inlier
                if( 0 < m_cClosureChecker.inliers( ) )
                {
                    for( LoopClosureChecker::EdgeDoubleMap::iterator itEdgeLoopClosure = m_cClosureChecker.closures( ).begin(); itEdgeLoopClosure != m_cClosureChecker.closures( ).end( ); itEdgeLoopClosure++ )
                    {
                        //ds get edge form
                        g2o::EdgeSE3* pEdgeLoopClosure = dynamic_cast< g2o::EdgeSE3* >( itEdgeLoopClosure->first );

                        if( m_dMaximumThresholdLoopClosing > itEdgeLoopClosure->second )
                        {
                            std::printf( "ADDING CLOSURE %06i to %06i WITH CHI2: %f\n", pEdgeLoopClosure->vertices( )[0]->id( )-1, pEdgeLoopClosure->vertices( )[1]->id( )-1, itEdgeLoopClosure->second );

                            //ds add to graph
                            m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
                            vecLoopClosures.push_back( CLoopClosureRaw( pEdgeLoopClosure->vertices( )[0]->id( )-1, pEdgeLoopClosure->vertices( )[1]->id( )-1, pEdgeLoopClosure ) );
                        }
                    }

                    //ds remove all candidate edges added so far
                    m_cBufferClosures.removeEdgeSet( m_cBufferClosures.edgeSet( ) );
                }
            }

            //ds update buffer
            m_cBufferClosures.updateList( p_uNumberOfClosedKeyFrames );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;

            //ds we always process all keyframes
            m_vecKeyFramesInGraph.push_back( pKeyFrame );
        }

        //ds if there were any closures accepted
        if( !vecLoopClosures.empty( ) )
        {
            std::printf( "[0][%06lu]<Cg2oOptimizer>(optimizeContinuous) optimizing loop closures: %lu/%u\n", p_uFrameCount, vecLoopClosures.size( ), uLoopClosuresTotal );

            //ds lock selected closures
            for( CLoopClosureRaw cEdgeLoopClosureRaw: vecLoopClosures )
            {
                const g2o::HyperGraph::VertexIDMap::iterator itReference( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( cEdgeLoopClosureRaw.uIDReference+1 ) );
                assert( itReference != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
                g2o::VertexSE3* pVertexReference = dynamic_cast< g2o::VertexSE3* >( itReference->second );
                pVertexReference->setFixed( true );
            }

            //ds save closure situation
            m_cOptimizerSparseTrajectoryOnly.save( ( strSaveFilePrefixLC + "_closed.g2o" ).c_str( ) );

            //ds timing
            const double dTimeStartSecondsLC = CLogger::getTimeSeconds( );
            const uint64_t uIterationsLC     = 1000;

            //ds always 1000 steps for loop closing!
            m_cOptimizerSparseTrajectoryOnly.initializeOptimization( );
            m_cOptimizerSparseTrajectoryOnly.optimize( uIterationsLC );

            //ds unlock added closures
            for( CLoopClosureRaw cEdgeLoopClosureRaw: vecLoopClosures )
            {
                //ds relax locked reference vertices again
                const g2o::HyperGraph::VertexIDMap::iterator itReference( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( cEdgeLoopClosureRaw.uIDReference+1 ) );
                assert( itReference != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
                g2o::VertexSE3* pVertexReference = dynamic_cast< g2o::VertexSE3* >( itReference->second );
                pVertexReference->setFixed( false );
            }

            //ds back propagate the trajectory only graph into the full one - BLOCKING key frames
            _backPropagateTrajectoryToFull( vecLoopClosures );

            m_cOptimizerSparseTrajectoryOnly.save( ( strSaveFilePrefixLC + "_closed_optimized.g2o" ).c_str( ) );
            std::printf( "[0][%06lu]<Cg2oOptimizer>(optimizeContinuous) optimization complete (total duration: %.2fs | iterations: %lu)\n",
                         p_uFrameCount, ( CLogger::getTimeSeconds( )-dTimeStartSecondsLC ), uIterationsLC );
        }
        else
        {
            std::printf( "[0][%06lu]<Cg2oOptimizer>(optimizeContinuous) DROPPED loop closures: %u\n", p_uFrameCount, uLoopClosuresTotal );
        }
    }

    //ds set landmarks - BLOCKING landmarks
    _loadLandmarksToGraph( uIDBeginLandmark, p_vecTranslationToG2o );

    //ds info
    std::vector< const CMeasurementLandmark* >::size_type uMeasurementsStoredPointXYZ    = 0;
    std::vector< const CMeasurementLandmark* >::size_type uMeasurementsStoredUVDepth     = 0;
    std::vector< const CMeasurementLandmark* >::size_type uMeasurementsStoredUVDisparity = 0;

    //ds if loop closed - we don't have to add the frames again, just the landmarks
    if( 0 < p_uNumberOfClosedKeyFrames )
    {
        //ds loop over the camera vertices vector
        for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < m_vecKeyFrames->size( ); ++uID )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds find the corresponding pose in the current graph
            const g2o::HyperGraph::VertexIDMap::iterator itPoseCurrent( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
            assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseCurrent );

            //ds get current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = dynamic_cast< g2o::VertexSE3* >( itPoseCurrent->second );
            assert( 0 != pVertexPoseCurrent );

            //ds always save acceleration data
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds if the key frame was closed
            if( !pKeyFrame->vecLoopClosures.empty( ) )
            {
                //ds for each closure
                for( const CKeyFrame::CMatchICP* pMatch: pKeyFrame->vecLoopClosures )
                {
                    //ds connect all closed landmarks
                    for( const CMatchCloud& cMatch: *pMatch->vecMatches )
                    {
                        //ds find the corresponding landmarks
                        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkQuery( m_cOptimizerSparse.vertices( ).find( cMatch.pPointQuery->uID ) );
                        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkReference( m_cOptimizerSparse.vertices( ).find( cMatch.pPointReference->uID ) );

                        //ds if query is in the graph
                        if( itLandmarkQuery != m_cOptimizerSparse.vertices( ).end( ) )
                        {
                            //ds consistency
                            assert( cMatch.pPointReference->uID == m_vecLandmarks->at( cMatch.pPointReference->uID )->uID );

                            //ds reference landmark
                            g2o::VertexPointXYZ* pVertexLandmarkReference = 0;

                            //ds check if the reference landmark is present in the graph
                            if( itLandmarkReference != m_cOptimizerSparse.vertices( ).end( ) )
                            {
                                //ds get it from the graph
                                pVertexLandmarkReference = dynamic_cast< g2o::VertexPointXYZ* >( itLandmarkReference->second );

                                //ds fix reference landmark
                                pVertexLandmarkReference->setFixed( true );

                                //ds set up the edge
                                g2o::EdgePointXYZ* pEdgeLandmarkClosure( new g2o::EdgePointXYZ( ) );

                                //ds set viewpoints and measurement
                                pEdgeLandmarkClosure->setVertex( 0, pVertexLandmarkReference );
                                pEdgeLandmarkClosure->setVertex( 1, itLandmarkQuery->second );
                                pEdgeLandmarkClosure->setMeasurement( Eigen::Vector3d::Zero( ) );
                                pEdgeLandmarkClosure->setInformation( m_matInformationLandmarkClosure );
                                pEdgeLandmarkClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                                //ds add to optimizer
                                m_cOptimizerSparse.addEdge( pEdgeLandmarkClosure );
                            }
                        }
                    }
                }
            }

            //ds we dont have to add the key frames as they have been added in the loop closing optimization before
        }
    }
    else
    {
        //ds loop over the camera vertices vector
        for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < m_vecKeyFrames->size( ); ++uID )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_uIDKeyFrameFrom, pKeyFrame, p_vecTranslationToG2o );

            //ds always save acceleration data
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;

            //ds we always process all keyframes
            m_vecKeyFramesInGraph.push_back( pKeyFrame );
        }
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", m_vecKeyFramesInGraph.front( )->uID, m_vecKeyFramesInGraph.back( )->uID );
    const std::string strSaveFilePrefix = chBuffer;
    m_cOptimizerSparse.save( ( strSaveFilePrefix + ".g2o" ).c_str( ) );

    std::printf( "[0][%06lu]<Cg2oOptimizer>(optimizeContinuous) optimizing [keyframes: %06lu-%06lu (%3lu) landmarks: %06lu-%06lu][measurements xyz: %3lu depth: %3lu disparity: %3lu] \n",
                 p_uFrameCount, p_uIDBeginKeyFrame, m_vecKeyFrames->back( )->uID, ( m_vecKeyFrames->back( )->uID-p_uIDBeginKeyFrame ),
                 uIDBeginLandmark, m_vecLandmarksInGraph.back( )->uID, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

    //ds timing
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );
    const uint64_t uIterations     = _optimizeUnLimited( m_cOptimizerSparse );

    //ds apply to structures
    _applyOptimizationToLandmarks( p_uFrameCount, uIDBeginLandmark, p_vecTranslationToG2o );
    _applyOptimizationToKeyFrames( p_uFrameCount, p_vecTranslationToG2o );
    _backPropagateTrajectoryToPure( );
    assert( m_vecKeyFramesInGraph.back( )->bIsOptimized );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strSaveFilePrefix + "_optimized.g2o" ).c_str( ) );
    std::printf( "[0][%06lu]<Cg2oOptimizer>(optimizeContinuous) optimization complete (total duration: %.2fs | iterations: %lu)\n", p_uFrameCount, ( CLogger::getTimeSeconds( )-dTimeStartSeconds ), uIterations );

    //ds timing
    m_dDurationTotalSecondsOptimization += ( CLogger::getTimeSeconds( )-dTimeStartSeconds );

    //ds info
    ++m_uOptimizations;
}

void Cg2oOptimizer::saveGraph( const UIDFrame& p_uFrameCount,
                               const UIDKeyFrame& p_uIDBeginKeyFrame,
                               const UIDKeyFrame& p_uNumberOfClosedKeyFrames,
                               const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds derive begin landmark within optimized chunk
    const UIDLandmark uIDBeginLandmark = m_vecKeyFrames->at( p_uIDBeginKeyFrame )->vecMeasurements.front( )->uID;

    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    assert( uIDBeginLandmark < m_vecKeyFrames->back( )->vecMeasurements.back( )->uID );

    //ds check if we have a loop closure (landmark free optimization)
    if( 0 < p_uNumberOfClosedKeyFrames )
    {
        //ds closure count
        uint32_t uLoopClosuresTotal = 0;

        //ds loop closure holder
        std::vector< CLoopClosureRaw > vecLoopClosures;

        //ds compute save file name
        char chBufferLC[256];
        std::snprintf( chBufferLC, 256, "g2o/local/keyframes_%06lu-%06lu", m_vecKeyFrames->at( p_uIDBeginKeyFrame )->uID, m_vecKeyFrames->back( )->uID );
        const std::string strSaveFilePrefixLC = chBufferLC;

        //ds loop over the camera vertices vector
        for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < m_vecKeyFrames->size( ); ++uID )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_uIDKeyFrameFrom, pKeyFrame, p_vecTranslationToG2o );

            //ds closed edges
            OptimizableGraph::EdgeSet vecLoopClosureEdges;

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                //ds accumulate the elements
                vecLoopClosureEdges.insert( _getEdgeLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure ) );
                ++uLoopClosuresTotal;

                //ds get loop closure
                //g2o::EdgeSE3* pEdgeLoopClosure = _getEdgeLoopClosure( pVertexPoseCurrent, pKeyFrame, pClosure );

                //ds add to graph
                //m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
                //vecLoopClosures.push_back( CLoopClosureRaw( pClosure->pKeyFrameReference->uID, pKeyFrame->uID, pEdgeLoopClosure ) );
            }

            //ds if there were any loop closures for this keyframe
            if( !vecLoopClosureEdges.empty( ) )
            {
                const g2o::HyperGraph::VertexIDMap::iterator itPoseCurrent( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );
                assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseCurrent );

                m_cBufferClosures.addEdgeSet( vecLoopClosureEdges );
                m_cBufferClosures.addVertex( dynamic_cast< g2o::VertexSE3* >( itPoseCurrent->second ) );
            }

            //ds evaluate loop closure window
            if( m_cBufferClosures.checkList( p_uNumberOfClosedKeyFrames ) )
            {
                //ds determine valid closures
                m_cClosureChecker.init( m_cBufferClosures.vertices( ), m_cBufferClosures.edgeSet( ), m_dMaximumThresholdLoopClosing );
                m_cClosureChecker.check( );

                std::printf( "WINDOW SIZE: %lu\n", p_uNumberOfClosedKeyFrames );
                std::printf( "INLIERS: %i\n", m_cClosureChecker.inliers( ) );

                //ds if at least one inlier
                if( 0 < m_cClosureChecker.inliers( ) )
                {
                    for( LoopClosureChecker::EdgeDoubleMap::iterator itEdgeLoopClosure = m_cClosureChecker.closures( ).begin(); itEdgeLoopClosure != m_cClosureChecker.closures( ).end( ); itEdgeLoopClosure++ )
                    {
                        //ds get edge form
                        g2o::EdgeSE3* pEdgeLoopClosure = dynamic_cast< g2o::EdgeSE3* >( itEdgeLoopClosure->first );

                        if( m_dMaximumThresholdLoopClosing > itEdgeLoopClosure->second )
                        {
                            std::printf( "ADDING CLOSURE %06i to %06i WITH CHI2: %f\n", pEdgeLoopClosure->vertices( )[0]->id( )-1, pEdgeLoopClosure->vertices( )[1]->id( )-1, itEdgeLoopClosure->second );

                            //ds add to graph
                            m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
                            vecLoopClosures.push_back( CLoopClosureRaw( pEdgeLoopClosure->vertices( )[0]->id( )-1, pEdgeLoopClosure->vertices( )[1]->id( )-1, pEdgeLoopClosure ) );
                        }
                    }

                    //ds remove all candidate edges added so far
                    m_cBufferClosures.removeEdgeSet( m_cBufferClosures.edgeSet( ) );
                }
            }

            //ds update buffer
            m_cBufferClosures.updateList( p_uNumberOfClosedKeyFrames );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;

            //ds we always process all keyframes
            m_vecKeyFramesInGraph.push_back( pKeyFrame );
        }

        //ds if there were any closures accepted
        if( !vecLoopClosures.empty( ) )
        {
            std::printf( "[0][%06lu]<Cg2oOptimizer>(saveGraph) optimizing loop closures: %lu/%u\n", p_uFrameCount, vecLoopClosures.size( ), uLoopClosuresTotal );

            //ds lock selected closures
            for( CLoopClosureRaw cEdgeLoopClosureRaw: vecLoopClosures )
            {
                const g2o::HyperGraph::VertexIDMap::iterator itReference( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( cEdgeLoopClosureRaw.uIDReference+1 ) );
                assert( itReference != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
                g2o::VertexSE3* pVertexReference = dynamic_cast< g2o::VertexSE3* >( itReference->second );
                pVertexReference->setFixed( true );
            }

            //ds save closure situation
            m_cOptimizerSparseTrajectoryOnly.save( ( strSaveFilePrefixLC + "_closed.g2o" ).c_str( ) );

            //ds timing
            const double dTimeStartSecondsLC = CLogger::getTimeSeconds( );
            const uint64_t uIterationsLC     = 1000;

            //ds unlock added closures
            for( CLoopClosureRaw cEdgeLoopClosureRaw: vecLoopClosures )
            {
                //ds relax locked reference vertices again
                const g2o::HyperGraph::VertexIDMap::iterator itReference( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( cEdgeLoopClosureRaw.uIDReference+1 ) );
                assert( itReference != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
                g2o::VertexSE3* pVertexReference = dynamic_cast< g2o::VertexSE3* >( itReference->second );
                pVertexReference->setFixed( false );
            }

            //ds back propagate the trajectory only graph into the full one - BLOCKING key frames
            _backPropagateTrajectoryToFull( vecLoopClosures );

            m_cOptimizerSparseTrajectoryOnly.save( ( strSaveFilePrefixLC + "_closed_optimized.g2o" ).c_str( ) );
            std::printf( "[0][%06lu]<Cg2oOptimizer>(saveGraph) optimization complete (total duration: %.2fs | iterations: %lu)\n",
                         p_uFrameCount, ( CLogger::getTimeSeconds( )-dTimeStartSecondsLC ), uIterationsLC );
        }
        else
        {
            std::printf( "[0][%06lu]<Cg2oOptimizer>(saveGraph) DROPPED loop closures: %u\n", p_uFrameCount, uLoopClosuresTotal );
        }
    }

    //ds set landmarks - BLOCKING landmarks
    _loadLandmarksToGraph( uIDBeginLandmark, p_vecTranslationToG2o );

    //ds info
    std::vector< const CMeasurementLandmark* >::size_type uMeasurementsStoredPointXYZ    = 0;
    std::vector< const CMeasurementLandmark* >::size_type uMeasurementsStoredUVDepth     = 0;
    std::vector< const CMeasurementLandmark* >::size_type uMeasurementsStoredUVDisparity = 0;

    //ds if loop closed - we don't have to add the frames again, just the landmarks
    if( 0 < p_uNumberOfClosedKeyFrames )
    {
        //ds loop over the camera vertices vector
        for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < m_vecKeyFrames->size( ); ++uID )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds find the corresponding pose in the current graph
            const g2o::HyperGraph::VertexIDMap::iterator itPoseCurrent( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
            assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseCurrent );

            //ds get current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = dynamic_cast< g2o::VertexSE3* >( itPoseCurrent->second );
            assert( 0 != pVertexPoseCurrent );

            //ds always save acceleration data
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds if the key frame was closed
            if( !pKeyFrame->vecLoopClosures.empty( ) )
            {
                //ds for each closure
                for( const CKeyFrame::CMatchICP* pMatch: pKeyFrame->vecLoopClosures )
                {
                    //ds connect all closed landmarks
                    for( const CMatchCloud& cMatch: *pMatch->vecMatches )
                    {
                        //ds find the corresponding landmarks
                        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkQuery( m_cOptimizerSparse.vertices( ).find( cMatch.pPointQuery->uID ) );
                        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkReference( m_cOptimizerSparse.vertices( ).find( cMatch.pPointReference->uID ) );

                        //ds if query is in the graph
                        if( itLandmarkQuery != m_cOptimizerSparse.vertices( ).end( ) )
                        {
                            //ds consistency
                            assert( cMatch.pPointReference->uID == m_vecLandmarks->at( cMatch.pPointReference->uID )->uID );

                            //ds reference landmark
                            g2o::VertexPointXYZ* pVertexLandmarkReference = 0;

                            //ds check if the reference landmark is present in the graph
                            if( itLandmarkReference != m_cOptimizerSparse.vertices( ).end( ) )
                            {
                                //ds get it from the graph
                                pVertexLandmarkReference = dynamic_cast< g2o::VertexPointXYZ* >( itLandmarkReference->second );

                                //ds fix reference landmark
                                pVertexLandmarkReference->setFixed( true );

                                //ds set up the edge
                                g2o::EdgePointXYZ* pEdgeLandmarkClosure( new g2o::EdgePointXYZ( ) );

                                //ds set viewpoints and measurement
                                pEdgeLandmarkClosure->setVertex( 0, pVertexLandmarkReference );
                                pEdgeLandmarkClosure->setVertex( 1, itLandmarkQuery->second );
                                pEdgeLandmarkClosure->setMeasurement( Eigen::Vector3d::Zero( ) );
                                pEdgeLandmarkClosure->setInformation( m_matInformationLandmarkClosure );
                                pEdgeLandmarkClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                                //ds add to optimizer
                                m_cOptimizerSparse.addEdge( pEdgeLandmarkClosure );
                            }
                        }
                    }
                }
            }

            //ds we dont have to add the key frames as they have been added in the loop closing optimization before
        }
    }
    else
    {
        //ds loop over the camera vertices vector
        for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < m_vecKeyFrames->size( ); ++uID )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds add current camera pose
            g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_uIDKeyFrameFrom, pKeyFrame, p_vecTranslationToG2o );

            //ds always save acceleration data
            m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

            //ds set landmark measurements
            _setLandmarkMeasurementsWORLD( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

            //ds update last
            m_uIDKeyFrameFrom = pKeyFrame->uID;

            //ds we always process all keyframes
            m_vecKeyFramesInGraph.push_back( pKeyFrame );
        }
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", m_vecKeyFramesInGraph.front( )->uID, m_vecKeyFramesInGraph.back( )->uID );
    const std::string strSaveFilePrefix = chBuffer;
    m_cOptimizerSparse.save( ( strSaveFilePrefix + ".g2o" ).c_str( ) );

    std::printf( "[0][%06lu]<Cg2oOptimizer>(saveGraph) optimizing [keyframes: %06lu-%06lu (%3lu) landmarks: %06lu-%06lu][measurements xyz: %3lu depth: %3lu disparity: %3lu] \n",
                 p_uFrameCount, p_uIDBeginKeyFrame, m_vecKeyFrames->back( )->uID, ( m_vecKeyFrames->back( )->uID-p_uIDBeginKeyFrame ),
                 uIDBeginLandmark, m_vecLandmarksInGraph.back( )->uID, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

    //ds timing
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );
    const uint64_t uIterations     = 0;

    //ds apply to structures
    _applyOptimizationToLandmarks( p_uFrameCount, uIDBeginLandmark, p_vecTranslationToG2o );
    _applyOptimizationToKeyFrames( p_uFrameCount, p_vecTranslationToG2o );
    _backPropagateTrajectoryToPure( );
    assert( m_vecKeyFramesInGraph.back( )->bIsOptimized );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strSaveFilePrefix + "_optimized.g2o" ).c_str( ) );
    std::printf( "[0][%06lu]<Cg2oOptimizer>(saveGraph) optimization complete (total duration: %.2fs | iterations: %lu)\n", p_uFrameCount, ( CLogger::getTimeSeconds( )-dTimeStartSeconds ), uIterations );

    //ds timing
    m_dDurationTotalSecondsOptimization += ( CLogger::getTimeSeconds( )-dTimeStartSeconds );

    //ds info
    ++m_uOptimizations;
}

void Cg2oOptimizer::clearFilesUNIX( ) const
{
    //ds directory handle
    DIR *pDirectory = opendir ( "g2o/local" );

    //ds try to open the directory
    if( 0 != pDirectory )
    {
        //ds file handle
        struct dirent *pFile;

        //ds delete all files
        while( ( pFile = readdir( pDirectory ) ) != NULL )
        {
            //ds validate filename
            if( '.' != pFile->d_name[0] )
            {
                //ds construct full filename and delete
                std::string strFile( "g2o/local/" );
                strFile += pFile->d_name;
                //std::printf( "<Cg2oOptimizer>(clearFiles) removing file: %s\n", strFile.c_str( ) );
                std::remove( strFile.c_str( ) );
            }
        }

        //ds close directory
        closedir( pDirectory );
    }
    else
    {
        //ds FATAL
        throw CExceptionLogfileTree( "g2o folder: g2o/local not set" );
    }
}

//ds manual loop closing
void Cg2oOptimizer::updateLoopClosuresFromKeyFrame( const UIDKeyFrame& p_uIDBeginKeyFrame,
                                                    const UIDKeyFrame& p_uIDEndKeyFrame,
                                                    const Eigen::Vector3d& p_vecTranslationToG2o )
{
    assert( p_uIDBeginKeyFrame < p_uIDEndKeyFrame );
    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFramesInGraph.size( ) );

    //ds update key frames in graph (null now) and g2o
    for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < p_uIDEndKeyFrame; ++uID )
    {
        assert( 0 != m_vecKeyFrames->at( uID ) );

        //ds only if updated
        if( 0 == m_vecKeyFramesInGraph[uID] )
        {
            //ds buffer keyframe
            CKeyFrame* pKeyFrame = m_vecKeyFrames->at( uID );

            //ds update internals
            m_vecKeyFramesInGraph[uID] = pKeyFrame;

            //ds update g2o: try to retrieve vertex
            const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

            //ds if we can update the vertex
            assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

            //ds extract the pose
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );

            assert( 0 != pVertex );

            //ds check if we got loop closures for this frame
            for( const CKeyFrame::CMatchICP* pClosure: pKeyFrame->vecLoopClosures )
            {
                assert( 0 != pClosure );

                //ds set the closure
                //_setLoopClosure( pVertex, pKeyFrame, pClosure, p_vecTranslationToG2o );

                //ds get loop closure
                g2o::EdgeSE3* pEdgeLoopClosure = _getEdgeLoopClosure( pVertex, pKeyFrame, pClosure );

                //ds add to graph
                m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgeLoopClosure );
            }
        }
    }
}

/*void Cg2oOptimizer::lockTrajectory( const std::vector< CKeyFrame* >::size_type& p_uIDBeginKeyFrame,
                                    const std::vector< CKeyFrame* >::size_type& p_uIDEndKeyFrame )
{
    assert( p_uIDBeginKeyFrame < p_uIDEndKeyFrame );
    assert( p_uIDBeginKeyFrame < m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFrames->size( ) );
    assert( p_uIDEndKeyFrame <= m_vecKeyFramesInGraph.size( ) );

    //ds lock all poses
    for( std::vector< CKeyFrame* >::size_type uID = p_uIDBeginKeyFrame; uID < p_uIDEndKeyFrame; ++uID )
    {
        //ds update g2o: try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( uID+m_uIDShift ) );

        //ds if we can update the vertex
        assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

        //ds extract the pose and lock it
        g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );
        pVertex->setFixed( true );

        std::cerr << "fixed keyframe: " << uID << std::endl;
    }

    //ds if feasiable
    if( 50 < m_vecKeyFramesInGraph.back( )->uID )
    {
        //ds lower limit
        const std::vector< CKeyFrame* >::size_type uIDStart = m_vecKeyFramesInGraph.back( )->uID-50;

        //ds increase stiffness of recent trajectory
        for( std::vector< CKeyFrame* >::size_type uID = m_vecKeyFramesInGraph.back( )->uID; uID > uIDStart; --uID )
        {
            //ds update g2o: try to retrieve vertex
            const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( uID+m_uIDShift ) );

            //ds if we can update the vertex
            assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

            //ds extract the pose and lock it
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itPose->second );

            assert( 0 != pVertex );

            //ds buffer connecting edge
            g2o::EdgeSE3* pEdgePoseFromTo = dynamic_cast< g2o::EdgeSE3* >( *( pVertex->edges( ).begin( ) ) );

            if( 0 != pEdgePoseFromTo )
            {
                //ds increase information matrix
                pEdgePoseFromTo->setInformation( static_cast< Eigen::Matrix< double, 6,6 > >( 100*pEdgePoseFromTo->information( ) ) );

                std::cerr << "strengthened keyframe: " << uID << std::endl;
            }
        }
    }
}*/

uint64_t Cg2oOptimizer::_optimizeUnLimited( g2o::SparseOptimizer& p_cOptimizer )
{
    //ds initialize optimization
    p_cOptimizer.initializeOptimization( );

    //ds initial optimization
    p_cOptimizer.optimize( 1 );

    //ds iterations count
    uint64_t uIterations = 1;

    //ds start optimization
    double dPreviousChi2 = 1.1*p_cOptimizer.chi2( );

    //ds start cycle and check if we have at least 1% chi improvement
    while( ( 0.99 > p_cOptimizer.chi2( )/dPreviousChi2 ) )
    {
        //ds update chi2
        dPreviousChi2 = p_cOptimizer.chi2( );

        //ds do ten iterations
        p_cOptimizer.optimize( 10 );
        uIterations += 10;
    }

    return uIterations;
}

g2o::EdgeSE3LinearAcceleration* Cg2oOptimizer::_getEdgeLinearAcceleration( g2o::VertexSE3* p_pVertexPose,
                                                                           const CLinearAccelerationIMU& p_vecLinearAccelerationNormalized ) const
{
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAcceleration = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAcceleration->setVertex( 0, p_pVertexPose );
    pEdgeLinearAcceleration->setMeasurement( p_vecLinearAccelerationNormalized );
    pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 5, 0, 0, 0, 5, 0, 0, 0, 5 };
    pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );

    //ds set robust kernel
    //pEdgeLinearAcceleration->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeLinearAcceleration;
}

g2o::EdgeSE3PointXYZ* Cg2oOptimizer::_getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                               const CPoint3DWORLD& p_vecPointXYZ,
                                               const double& p_dInformationFactor ) const
{
    g2o::EdgeSE3PointXYZ* pEdgePointXYZ( new g2o::EdgeSE3PointXYZ( ) );

    //ds triangulated 3d point (uncalibrated)
    pEdgePointXYZ->setVertex( 0, p_pVertexPose );
    pEdgePointXYZ->setVertex( 1, p_pVertexLandmark );
    pEdgePointXYZ->setMeasurement( p_vecPointXYZ );
    pEdgePointXYZ->setParameterId( 0, EG2OParameterID::eWORLD );

    //ds the closer to the camera the point is the more meaningful is the measurement
    //const double dInformationStrengthXYZ( m_dMaximumReliableDepth/( 1+pMeasurementLandmark->vecPointXYZ.z( ) ) );
    const double arrInformationMatrixXYZ[9] = { p_dInformationFactor*1000, 0, 0, 0, p_dInformationFactor*1000, 0, 0, 0, p_dInformationFactor*1000 };
    pEdgePointXYZ->setInformation( g2o::Matrix3D( arrInformationMatrixXYZ ) );

    //ds set robust kernel
    //pEdgePointXYZ->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgePointXYZ;
}

g2o::EdgeSE3PointXYZDepth* Cg2oOptimizer::_getEdgeUVDepthLEFT( g2o::VertexSE3* p_pVertexPose,
                                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                                               const CMeasurementLandmark* p_pMeasurement,
                                                               const double& p_dInformationFactor ) const
{
    //ds projected depth
    g2o::EdgeSE3PointXYZDepth* pEdgeProjectedDepth( new g2o::EdgeSE3PointXYZDepth( ) );

    pEdgeProjectedDepth->setVertex( 0, p_pVertexPose );
    pEdgeProjectedDepth->setVertex( 1, p_pVertexLandmark );
    pEdgeProjectedDepth->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, p_pMeasurement->vecPointXYZLEFT.z( ) ) );
    pEdgeProjectedDepth->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDepth( m_dMaximumReliableDepth/dDepthMeters );
    const double arrInformationMatrixDepth[9] = { p_dInformationFactor, 0, 0, 0, p_dInformationFactor, 0, 0, 0, p_dInformationFactor*100 };
    pEdgeProjectedDepth->setInformation( g2o::Matrix3D( arrInformationMatrixDepth ) );

    //ds set robust kernel
    //pEdgeProjectedDepth->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeProjectedDepth;
}

g2o::EdgeSE3PointXYZDisparity* Cg2oOptimizer::_getEdgeUVDisparityLEFT( g2o::VertexSE3* p_pVertexPose,
                                                           g2o::VertexPointXYZ* p_pVertexLandmark,
                                                           const double& p_dDisparityPixels,
                                                           const CMeasurementLandmark* p_pMeasurement,
                                                           const double& p_dFxPixels,
                                                           const double& p_dBaselineMeters,
                                                           const double& p_dInformationFactor ) const
{
    //ds disparity
    g2o::EdgeSE3PointXYZDisparity* pEdgeDisparity( new g2o::EdgeSE3PointXYZDisparity( ) );

    const double dDisparityNormalized( p_dDisparityPixels/( p_dFxPixels*p_dBaselineMeters ) );
    pEdgeDisparity->setVertex( 0, p_pVertexPose );
    pEdgeDisparity->setVertex( 1, p_pVertexLandmark );
    pEdgeDisparity->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, dDisparityNormalized ) );
    pEdgeDisparity->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDisparity( std::abs( dDisparity )+1.0 );
    const double arrInformationMatrixDisparity[9] = { p_dInformationFactor, 0, 0, 0, p_dInformationFactor, 0, 0, 0, p_dInformationFactor*1000 };
    pEdgeDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

    //ds set robust kernel
    //pEdgeDisparity->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    return pEdgeDisparity;
}

g2o::EdgeSE3* Cg2oOptimizer::_getEdgeLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure )
{
    /*ds find the corresponding pose in the current graph
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparse.vertices( ).find( p_pClosure->pKeyFrameReference->uID+m_uIDShift ) );

    //ds has to be found
    assert( itClosure != m_cOptimizerSparse.vertices( ).end( ) );

    //ds pose to close
    g2o::VertexSE3* pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexClosure );
    pEdgeLoopClosure->setVertex( 1, p_pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( p_pClosure->matTransformationToClosure );
    pEdgeLoopClosure->setInformation( m_matInformationLoopClosure );*/

    //ds find the corresponding poses in the current graph
    const g2o::HyperGraph::VertexIDMap::iterator itCurrent( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrameCurrent->uID+1 ) );
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( p_pClosure->pKeyFrameReference->uID+1 ) );

    //ds have to be found
    assert( itCurrent != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );
    assert( itClosure != m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) );

    //ds buffer poses
    g2o::VertexSE3* pVertexPoseClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );
    g2o::VertexSE3* pVertexPoseCurrent = dynamic_cast< g2o::VertexSE3* >( itCurrent->second );

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexPoseClosure );
    pEdgeLoopClosure->setVertex( 1, pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( p_pClosure->matTransformationToClosure );
    pEdgeLoopClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    //ds compute error
    //const Eigen::Vector3d verTranslationDifference( ( pVertexPoseClosure->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) ).translation( )-p_pClosure->matTransformationToClosure.translation( ) );

    /*ds check information matrix - check if loop closure is effective or not
    if( 5.0 < std::fabs( verTranslationDifference.z( ) ) )
    {
        //ds full information
        pEdgeLoopClosure->setInformation( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*pEdgeLoopClosure->information( ) ) );
    }
    else
    {*/
        //ds damp z information
        pEdgeLoopClosure->setInformation( _getInformationNoZ( static_cast< Eigen::Matrix< double, 6, 6 > >( 10*pEdgeLoopClosure->information( ) ) ) );
        //pEdgeLoopClosure->setInformation( Eigen::Matrix< double, 6, 6 >::Zero( ) );
    //}

    return pEdgeLoopClosure;
}

void Cg2oOptimizer::_loadLandmarksToGraph( const UIDLandmark& p_uIDLandmark, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds info
    std::vector< CLandmark* >::size_type uDroppedLandmarksInvalid    = 0;
    std::vector< CLandmark* >::size_type uDroppedLandmarksKeyFraming = 0;
    std::vector< CLandmark* >::size_type uLandmarksAlreadyInGraph    = 0;

    assert( p_uIDLandmark < m_vecLandmarks->size( ) );

    //ds add landmarks
    for( std::vector< CLandmark* >::size_type uID = p_uIDLandmark; uID < m_vecLandmarks->size( ); ++uID )
    {
        //ds buffer landmark
        CLandmark* pLandmark = m_vecLandmarks->at( uID );

        assert( 0 != pLandmark );

        //ds check if optimiziation criterias are met
        if( pLandmark->bIsOptimal )
        {
            //ds landmark has to be present in 2 keyframes
            if( 1 < pLandmark->uNumberOfKeyFramePresences )
            {
                //ds updated estimate
                const CPoint3DWORLD vecPosition( pLandmark->vecPointXYZOptimized+p_vecTranslationToG2o );

                //ds if sane TODO make redundant
                if( m_dSanePositionThresholdL2 > vecPosition.squaredNorm( ) )
                {
                    //ds try to retrieve vertex
                    const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

                    //ds if found
                    if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
                    {
                        //ds extract and update the estimate
                        g2o::VertexPointXYZ* pVertex = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );
                        pVertex->setEstimate( vecPosition );

                        ++uLandmarksAlreadyInGraph;
                    }
                    else
                    {
                        //ds set landmark vertex
                        g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
                        pVertexLandmark->setEstimate( vecPosition );
                        pVertexLandmark->setId( pLandmark->uID );

                        //ds add vertex to optimizer
                        m_cOptimizerSparse.addVertex( pVertexLandmark );
                        m_vecLandmarksInGraph.push_back( pLandmark );
                    }
                }
            }
            else
            {
                ++uDroppedLandmarksKeyFraming;
            }
        }
        else
        {
            ++uDroppedLandmarksInvalid;
        }
    }

    //std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) dropped landmarks due to invalid optimization: %lu (%4.2f)\n", uDroppedLandmarksInvalid, static_cast< double >( uDroppedLandmarksInvalid )/p_vecChunkLandmarks.size( ) );
    //std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) dropped landmarks due to missing key frame presence: %lu (%4.2f)\n", uDroppedLandmarksKeyFraming, static_cast< double >( uDroppedLandmarksKeyFraming )/p_vecChunkLandmarks.size( ) );
    //std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) landmarks already present in graph: %lu\n", uLandmarksAlreadyInGraph );
}

g2o::VertexSE3* Cg2oOptimizer::_setAndgetPose( const UIDKeyFrame& p_uIDKeyFrameFrom, CKeyFrame* pKeyFrameCurrent, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds compose transformation matrix
    Eigen::Isometry3d matTransformationLEFTtoWORLD = pKeyFrameCurrent->matTransformationLEFTtoWORLD;
    matTransformationLEFTtoWORLD.translation( ) += p_vecTranslationToG2o;

    //ds add current camera pose
    g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
    pVertexPoseCurrent->setEstimate( matTransformationLEFTtoWORLD );
    pVertexPoseCurrent->setId( pKeyFrameCurrent->uID+m_uIDShift );
    m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

    //ds for trajectory only as well
    g2o::VertexSE3* pVertexPoseCurrentTrajectoryOnly = new g2o::VertexSE3( );
    pVertexPoseCurrentTrajectoryOnly->setEstimate( matTransformationLEFTtoWORLD );
    pVertexPoseCurrentTrajectoryOnly->setId( pKeyFrameCurrent->uID+1 );
    m_cOptimizerSparseTrajectoryOnly.addVertex( pVertexPoseCurrentTrajectoryOnly );

    //ds set up the edge to connect the poses
    g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

    //ds set viewpoints and measurement
    const g2o::HyperGraph::VertexIDMap::iterator itPoseFrom = m_cOptimizerSparse.vertices( ).find( p_uIDKeyFrameFrom+m_uIDShift );
    assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseFrom );
    g2o::VertexSE3* pVertexPoseFrom = dynamic_cast< g2o::VertexSE3* >( itPoseFrom->second );
    pEdgePoseFromTo->setVertex( 0, pVertexPoseFrom );
    pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
    pEdgePoseFromTo->setMeasurement( pVertexPoseFrom->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

    //ds compute information value based on absolute distance between keyframes
    const double dInformationFactor = 1.0/(1.0+pEdgePoseFromTo->measurement( ).translation( ).squaredNorm( ) );

    //ds new information matrix
    Eigen::Matrix< double, 6, 6 > matInformation( m_matInformationPose );
    matInformation.block< 3,3 >(0,0) *= dInformationFactor;

    //ds set specific information matrix (far connections between keyframes are less rigid)
    pEdgePoseFromTo->setInformation( matInformation );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

    //ds find the corresponding pose in the trajectory only graph
    const g2o::HyperGraph::VertexIDMap::iterator itPoseFromTrajectoryOnly( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( p_uIDKeyFrameFrom+1 ) );
    assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseFromTrajectoryOnly );

    //ds trajectory only
    g2o::EdgeSE3* pEdgePoseFromToTrajectoryOnly = new g2o::EdgeSE3( );
    pEdgePoseFromToTrajectoryOnly->setVertex( 0, dynamic_cast< g2o::VertexSE3* >( itPoseFromTrajectoryOnly->second ) );
    pEdgePoseFromToTrajectoryOnly->setVertex( 1, pVertexPoseCurrentTrajectoryOnly );
    pEdgePoseFromToTrajectoryOnly->setMeasurement( pEdgePoseFromTo->measurement( ) );
    Eigen::Matrix< double, 6, 6 > matInformationTrajectoryOnly( pEdgePoseFromToTrajectoryOnly->information( ) );
    matInformationTrajectoryOnly.block< 3,3 >(0,0) *= dInformationFactor;
    pEdgePoseFromToTrajectoryOnly->setInformation( matInformationTrajectoryOnly );
    m_cOptimizerSparseTrajectoryOnly.addEdge( pEdgePoseFromToTrajectoryOnly );

    //ds add to control structure
    m_vecPoseEdges.push_back( pEdgePoseFromTo );
    m_mapEdgeIDs.insert( std::make_pair( pKeyFrameCurrent->uID, std::make_pair( pEdgePoseFromTo, pEdgePoseFromToTrajectoryOnly ) ) );

    return pVertexPoseCurrent;
}

void Cg2oOptimizer::_setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent, const CKeyFrame::CMatchICP* p_pClosure, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds find the corresponding pose in the current graph
    assert( 0 != p_pClosure->pKeyFrameReference );
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparse.vertices( ).find( p_pClosure->pKeyFrameReference->uID+m_uIDShift ) );

    //ds has to be found
    assert( itClosure != m_cOptimizerSparse.vertices( ).end( ) );

    //ds pose to close
    g2o::VertexSE3* pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );

    /*ds if found
    if( itClosure != m_cOptimizerSparse.vertices( ).end( ) )
    {
        //ds extract the pose
        pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );
    }
    else
    {
        //ds compose transformation matrix
        Eigen::Isometry3d matTransformationLEFTtoWORLD = p_pClosure->pKeyFrameReference->matTransformationLEFTtoWORLD;
        matTransformationLEFTtoWORLD.translation( ) += p_vecTranslationToG2o;

        //ds we have to add the pose temporarily and fix it
        pVertexClosure = new g2o::VertexSE3( );
        pVertexClosure->setEstimate( matTransformationLEFTtoWORLD );
        pVertexClosure->setId( p_pClosure->pKeyFrameReference->uID+m_uIDShift );
        m_cOptimizerSparse.addVertex( pVertexClosure );
    }*/

    //ds fix reference
    pVertexClosure->setFixed( true );

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexClosure );
    pEdgeLoopClosure->setVertex( 1, p_pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( p_pClosure->matTransformationToClosure );
    pEdgeLoopClosure->setInformation( m_matInformationLoopClosure );

    //ds kernelize loop closing
    //pEdgeLoopClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgeLoopClosure );

    /*ds connect all closed landmarks
    for( const CMatchCloud& cMatch: *p_pClosure->vecMatches )
    {
        //ds find the corresponding landmarks
        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkQuery( m_cOptimizerSparse.vertices( ).find( cMatch.cPointQuery.uID ) );
        const g2o::HyperGraph::VertexIDMap::iterator itLandmarkReference( m_cOptimizerSparse.vertices( ).find( cMatch.cPointReference.uID ) );

        //ds if query is in the graph
        if( itLandmarkQuery != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds consistency
            assert( cMatch.cPointReference.uID == m_vecLandmarks->at( cMatch.cPointReference.uID )->uID );

            //ds reference landmark
            g2o::VertexPointXYZ* pVertexLandmarkReference = 0;

            //ds check if the reference landmark is present in the graph
            if( itLandmarkReference != m_cOptimizerSparse.vertices( ).end( ) )
            {
                //ds get it from the graph
                pVertexLandmarkReference = dynamic_cast< g2o::VertexPointXYZ* >( itLandmarkReference->second );

                //ds fix reference landmark
                pVertexLandmarkReference->setFixed( true );

                //ds set up the edge
                g2o::EdgePointXYZ* pEdgeLandmarkClosure( new g2o::EdgePointXYZ( ) );

                //ds set viewpoints and measurement
                pEdgeLandmarkClosure->setVertex( 0, itLandmarkQuery->second );
                pEdgeLandmarkClosure->setVertex( 1, pVertexLandmarkReference );
                pEdgeLandmarkClosure->setMeasurement( Eigen::Vector3d::Zero( ) );
                pEdgeLandmarkClosure->setInformation( m_matInformationLandmarkClosure );
                pEdgeLandmarkClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                //ds add to optimizer
                m_cOptimizerSparse.addEdge( pEdgeLandmarkClosure );
            }
        }
    }*/
}

void Cg2oOptimizer::_setLandmarkMeasurementsWORLD( g2o::VertexSE3* p_pVertexPoseCurrent,
                                      const CKeyFrame* pKeyFrameCurrent,
                                      std::vector< const CMeasurementLandmark* >::size_type& p_uMeasurementsStoredPointXYZ,
                                      std::vector< const CMeasurementLandmark* >::size_type& p_uMeasurementsStoredUVDepth,
                                      std::vector< const CMeasurementLandmark* >::size_type& p_uMeasurementsStoredUVDisparity )
{
    //ds check visible landmarks and add the edges for the current pose
    for( const CMeasurementLandmark* pMeasurementLandmark: pKeyFrameCurrent->vecMeasurements )
    {
        //ds find the corresponding landmark
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pMeasurementLandmark->uID ) );

        //ds if found
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
            g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

            //ds get optimized landmark into current pose
            const CPoint3DCAMERA vecPointLEFTEstimate( p_pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

            //ds current distance situation
            const double dDistanceL2Absolute = pMeasurementLandmark->vecPointXYZLEFT.squaredNorm( );
            const double dDistanceL2Relative = vecPointLEFTEstimate.squaredNorm( )/dDistanceL2Absolute;

            //ds check if the measurement is sufficiently consistent
            if( 0.75 < dDistanceL2Relative && 1.25 > dDistanceL2Relative )
            {
                //ds the closer the more reliable (a depth less than 1 meter actually increases the informational value)
                const double dInformationFactor  = 1.0/pMeasurementLandmark->vecPointXYZLEFT.z( );

                //ds maximum depth to produce a reliable XYZ estimate
                if( m_dMaximumReliableDepthForPointXYZL2 > dDistanceL2Absolute )
                {
                    //ds retrieve the edge
                    g2o::EdgeSE3PointXYZ* pEdgePointXYZ = _getEdgePointXYZ( p_pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark->vecPointXYZLEFT, dInformationFactor );

                    //ds add the edge to the graph
                    m_cOptimizerSparse.addEdge( pEdgePointXYZ );
                    ++p_uMeasurementsStoredPointXYZ;
                }

                //ds still good enough for uvdepth
                else if( m_dMaximumReliableDepthForUVDepthL2 > dDistanceL2Absolute )
                {
                    //ds get the edge
                    g2o::EdgeSE3PointXYZDepth* pEdgeUVDepth = _getEdgeUVDepthLEFT( p_pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark, dInformationFactor );

                    //ds projected depth (LEFT camera)
                    m_cOptimizerSparse.addEdge( pEdgeUVDepth );
                    ++p_uMeasurementsStoredUVDepth;
                }

                //ds go with disparity
                else if( m_dMaximumReliableDepthForUVDisparityL2 > dDistanceL2Absolute )
                {
                    //ds current disparity
                    const double dDisparity = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;

                    //ds at least 2 pixels stereo distance
                    if( 1.0 < dDisparity )
                    {
                        //ds get a disparity edge
                        g2o::EdgeSE3PointXYZDisparity* pEdgeUVDisparity( _getEdgeUVDisparityLEFT( p_pVertexPoseCurrent,
                                                                                                  pVertexLandmark,
                                                                                                  dDisparity,
                                                                                                  pMeasurementLandmark,
                                                                                                  m_pCameraSTEREO->m_pCameraLEFT->m_dFxP,
                                                                                                  m_pCameraSTEREO->m_dBaselineMeters,
                                                                                                  dInformationFactor ) );

                        //ds disparity (LEFT camera)
                        m_cOptimizerSparse.addEdge( pEdgeUVDisparity );
                        ++p_uMeasurementsStoredUVDisparity;
                    }
                }
            }
            else
            {
                //std::printf( "<Cg2oOptimizer>(_setLandmarkMeasurementsWORLD) ignoring malformed landmark [%06lu] depth difference: %f\n", pMeasurementLandmark->uID, dDifferenceDepth );
            }
        }
    }
}

void Cg2oOptimizer::_applyOptimizationToLandmarks( const UIDFrame& p_uFrameCount, const UIDLandmark& p_uIDLandmark, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    std::vector< CLandmark* >::size_type uNumberOfErasedLandmarks = 0;

    //ds update landmarks
    for( CLandmark* pLandmark: m_vecLandmarksInGraph )
    {
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

        //ds must be present
        //assert( itLandmark != m_cOptimizerSparse.vertices( ).end( ) );
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds get the vertex
            g2o::VertexPointXYZ* pVertex = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

            //ds check if sane
            if( m_dSanePositionThresholdL2 > pVertex->estimate( ).squaredNorm( ) )
            {
                //ds update position
                pLandmark->vecPointXYZOptimized = pVertex->estimate( )-p_vecTranslationToG2o;
            }
            else
            {
                //ds reset g2o instance
                ++uNumberOfErasedLandmarks;

                //ds detach the vertex and its edges - edges first
                for( g2o::HyperGraph::EdgeSet::const_iterator itEdge = pVertex->edges( ).begin( ); itEdge != pVertex->edges( ).end( ); ++itEdge )
                {
                    m_cOptimizerSparse.removeEdge( *itEdge );
                }

                //ds remove the vertex
                m_cOptimizerSparse.removeVertex( pVertex );
            }
        }
    }

    if( 0 < uNumberOfErasedLandmarks )
    {
        std::printf( "[%06lu]<Cg2oOptimizer>(_applyOptimization) erased badly optimized landmarks: %lu\n", p_uFrameCount, uNumberOfErasedLandmarks );
    }
}

void Cg2oOptimizer::_applyOptimizationToKeyFrames( const UIDFrame& p_uFrameCount, const Eigen::Vector3d& p_vecTranslationToG2o )
{
    //ds update poses
    for( CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds find vertex in graph
        //ds try to retrieve vertex
        const g2o::HyperGraph::VertexIDMap::iterator itKeyFrame( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds has to be found
        if( itKeyFrame != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds extract and update the keyframe
            g2o::VertexSE3* pVertex = dynamic_cast< g2o::VertexSE3* >( itKeyFrame->second );
            pKeyFrame->matTransformationLEFTtoWORLD                = pVertex->estimate( );
            pKeyFrame->matTransformationLEFTtoWORLD.translation( ) -= p_vecTranslationToG2o;
            pKeyFrame->bIsOptimized                                = true;

            //ds relax the vertex again for next loop-closing optimization
            //pVertex->setFixed( false );
        }
        else
        {
            std::printf( "[%06lu]<Cg2oOptimizer>(_applyOptimization) caught invalid key frame ID: %lu\n", p_uFrameCount, pKeyFrame->uID );
        }
    }
}

const Eigen::Matrix< double, 6, 6 > Cg2oOptimizer::_getInformationNoZ( const Eigen::Matrix< double, 6, 6 >& p_matInformationIN ) const
{
    Eigen::Matrix< double, 6, 6 > matInformationOUT( p_matInformationIN );

    //ds lower z by a factor
    matInformationOUT(2,2) /= 100.0;

    return matInformationOUT;
}

void Cg2oOptimizer::_backPropagateTrajectoryToFull( const std::vector< CLoopClosureRaw >& p_vecClosures )
{
    //ds back propagate the trajectory only graph into the full one
    for( const CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds search vertices
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
        const g2o::HyperGraph::VertexIDMap::iterator itPoseTrajectoryOnly( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );

        //ds must exist
        assert( m_cOptimizerSparse.vertices( ).end( )               != itPose );
        assert( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) != itPoseTrajectoryOnly );

        //ds //ds update estimate
        g2o::VertexSE3* pVertexPose               = dynamic_cast< g2o::VertexSE3* >( itPose->second );
        g2o::VertexSE3* pVertexPoseTrajectoryOnly = dynamic_cast< g2o::VertexSE3* >( itPoseTrajectoryOnly->second );
        pVertexPose->setEstimate( pVertexPoseTrajectoryOnly->estimate( ) );

        //ds fetch pose egdes
        const std::pair< g2o::EdgeSE3*, g2o::EdgeSE3* > prEdges( m_mapEdgeIDs[pKeyFrame->uID] );

        //ds update connecting edge measurement
        g2o::EdgeSE3* pEdge               = dynamic_cast< g2o::EdgeSE3* >( *pVertexPose->edges( ).find( prEdges.first ) );
        g2o::EdgeSE3* pEdgeTrajectoryOnly = dynamic_cast< g2o::EdgeSE3* >( *pVertexPoseTrajectoryOnly->edges( ).find( prEdges.second ) );
        pEdge->setMeasurement( pEdgeTrajectoryOnly->measurement( ) );
    }

    //ds back propagate loop closures to full graph
    for( CLoopClosureRaw cLoopClosure: p_vecClosures )
    {
        //ds search vertices in full graph
        const g2o::HyperGraph::VertexIDMap::iterator itPoseReference( m_cOptimizerSparse.vertices( ).find( cLoopClosure.uIDReference+m_uIDShift ) );
        const g2o::HyperGraph::VertexIDMap::iterator itPoseQuery( m_cOptimizerSparse.vertices( ).find( cLoopClosure.uIDQuery+m_uIDShift ) );

        //ds must exist
        assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseReference );
        assert( m_cOptimizerSparse.vertices( ).end( ) != itPoseQuery );

        //ds find edge in optimized graph
        const g2o::HyperGraph::EdgeSet::iterator itEdgeLoopClosure( m_cOptimizerSparseTrajectoryOnly.edges( ).find( cLoopClosure.pEdgeLoopClosureReferenceToQuery ) );
        assert( m_cOptimizerSparseTrajectoryOnly.edges( ).end( ) != itEdgeLoopClosure );
        g2o::EdgeSE3* pEdgeLoopClosureInGraph = dynamic_cast< g2o::EdgeSE3* >( *itEdgeLoopClosure );

        //ds set loop closure edge for full graph
        g2o::EdgeSE3* pEdgeLoopClosure = new g2o::EdgeSE3( );
        pEdgeLoopClosure->setVertex( 0, dynamic_cast< g2o::VertexSE3* >( itPoseReference->second ) );
        pEdgeLoopClosure->setVertex( 1, dynamic_cast< g2o::VertexSE3* >( itPoseQuery->second ) );
        pEdgeLoopClosure->setMeasurement( pEdgeLoopClosureInGraph->measurement( ) );
        pEdgeLoopClosure->setInformation( m_matInformationLoopClosure );
        m_cOptimizerSparse.addEdge( pEdgeLoopClosure );
    }
}

void Cg2oOptimizer::_backPropagateTrajectoryToPure( )
{
    //ds back propagate the trajectory only graph into the full one
    for( const CKeyFrame* pKeyFrame: m_vecKeyFramesInGraph )
    {
        //ds search vertices
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );
        const g2o::HyperGraph::VertexIDMap::iterator itPoseTrajectoryOnly( m_cOptimizerSparseTrajectoryOnly.vertices( ).find( pKeyFrame->uID+1 ) );

        //ds must exist (except for final optimization) - check if not
        if( ( m_cOptimizerSparse.vertices( ).end( )               == itPose               ) ||
            ( m_cOptimizerSparseTrajectoryOnly.vertices( ).end( ) == itPoseTrajectoryOnly ) )
        {
            //ds back propagation failed
            std::printf( "<Cg2oOptimizer>(_backPropagateTrajectoryToPure) back propagation to trajectory only failed for frame ID: %lu\n", pKeyFrame->uID );
            return;
        }

        //ds //ds update estimate
        g2o::VertexSE3* pVertexPose               = dynamic_cast< g2o::VertexSE3* >( itPose->second );
        g2o::VertexSE3* pVertexPoseTrajectoryOnly = dynamic_cast< g2o::VertexSE3* >( itPoseTrajectoryOnly->second );
        pVertexPoseTrajectoryOnly->setEstimate( pVertexPose->estimate( ) );

        //ds fetch pose egdes
        const std::pair< g2o::EdgeSE3*, g2o::EdgeSE3* > prEdges( m_mapEdgeIDs[pKeyFrame->uID] );

        //ds update connecting edge measurement
        g2o::EdgeSE3* pEdge               = dynamic_cast< g2o::EdgeSE3* >( *pVertexPose->edges( ).find( prEdges.first ) );
        g2o::EdgeSE3* pEdgeTrajectoryOnly = dynamic_cast< g2o::EdgeSE3* >( *pVertexPoseTrajectoryOnly->edges( ).find( prEdges.second ) );
        pEdgeTrajectoryOnly->setMeasurement( pEdge->measurement( ) );
    }
}
