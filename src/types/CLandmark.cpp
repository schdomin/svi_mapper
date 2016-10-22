#include "CLandmark.h"
#include "utility/CWrapperOpenCV.h"
#include "vision/CMiniVisionToolbox.h"
#include <fstream>
#include <random>
#include <bitset>

//#define NUMBER_OF_NOISY_BITS 26



CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptorLEFT,
           const CDescriptor& p_matDescriptorRIGHT,
           const double& p_dKeyPointSize,
           const cv::Point2d& p_ptUVLEFT,
           const cv::Point2d& p_ptUVRIGHT,
           const CPoint3DCAMERA& p_vecPointXYZLEFT,
           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
           const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
           const MatrixProjection& p_matProjectionLEFT,
           const MatrixProjection& p_matProjectionRIGHT,
           const MatrixProjection& p_matProjectionWORLDtoLEFT,
           const MatrixProjection& p_matProjectionWORLDtoRIGHT,
           const UIDFrame& p_uIDFrame ): uID( p_uID ),
                                         matDescriptorReferenceLEFT( p_matDescriptorLEFT ),
                                         matDescriptorReferenceRIGHT( p_matDescriptorRIGHT ),
                                         dKeyPointSize( p_dKeyPointSize ),
                                         uIDFrameAtCreation( p_uIDFrame ),
                                         vecPointXYZInitial( p_matTransformationLEFTtoWORLD*p_vecPointXYZLEFT ),
                                         vecPointXYZOptimized( vecPointXYZInitial ),
                                         vecUVReferenceLEFT( p_ptUVLEFT.x, p_ptUVLEFT.y, 1.0 ),
                                         vecPointXYZMean( vecPointXYZInitial ),
                                         m_matProjectionLEFT( p_matProjectionLEFT ),
                                         m_matProjectionRIGHT( p_matProjectionRIGHT )
{
    vecDescriptorsLEFT.clear( );
    vecDescriptorsRIGHT.clear( );
    //vecDescriptorsLEFTNoisy.clear( );
    m_vecMeasurements.clear( );

    //ds construct filestring and open dump file
    //char chBuffer[256];
    //std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks/landmark%06lu.txt", uID );
    //m_pFilePositionOptimization = std::fopen( chBuffer, "w" );

    //assert( 0 != m_pFilePositionOptimization );

    //ds dump file format
    //std::fprintf( m_pFilePositionOptimization, "ID_FRAME | ID_LANDMARK | ITERATION MEASUREMENTS INLIERS | ERROR_ARSS | DELTA_XYZ |      X      Y      Z\n" );

    //ds add this position
    addMeasurement( p_uIDFrame,
                    p_ptUVLEFT,
                    p_ptUVRIGHT,
                    p_matDescriptorLEFT,
                    p_matDescriptorRIGHT,
                    p_vecPointXYZLEFT,
                    p_matTransformationLEFTtoWORLD,
                    p_matTransformationWORLDtoLEFT,
                    p_matProjectionWORLDtoLEFT,
                    p_matProjectionWORLDtoRIGHT );
}

CLandmark::~CLandmark( )
{
    //ds close file
    //std::fclose( m_pFilePositionOptimization );

    //ds free positions
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        if( 0 != pMeasurement )
        {
            delete pMeasurement;
        }
    }
}

void CLandmark::addMeasurement( const UIDFrame& p_uFrame,
                                const cv::Point2d& p_ptUVLEFT,
                                const cv::Point2d& p_ptUVRIGHT,
                                const CDescriptor& p_matDescriptorLEFT,
                                const CDescriptor& p_matDescriptorRIGHT,
                                const CPoint3DCAMERA& p_vecXYZLEFT,
                                const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                const MatrixProjection& p_matProjectionWORLDtoRIGHT )
{
    //ds input validation
    assert( p_ptUVLEFT.y == p_ptUVRIGHT.y );
    assert( 0 < p_vecXYZLEFT.z( ) );

    //ds probability counting
    const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecDescriptorLEFT( CWrapperOpenCV::getDescriptorVector< double >( p_matDescriptorLEFT ) );

    //ds if we have at least one descriptor: TODO refactor, this if case is unecessary expensive
    if( 0 < vecDescriptorsLEFT.size( ) )
    {
        //ds buffer last descriptor
        const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecDescriptorLEFTLAST( CWrapperOpenCV::getDescriptorVector< double >( vecDescriptorsLEFT.back( ) ) );

        //ds bit volatility counting
        for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
        {
            //ds if the bit matches the previous one
            if( vecDescriptorLEFTLAST[u] == vecDescriptorLEFT[u] )
            {
                ++m_vecBitPermanenceActive[u];
            }
            else
            {
                //ds reset active count
                m_vecBitPermanenceActive[u] = 0;
            }

            //ds check if we have to update the count
            if( m_vecBitPermanenceMaximum[u] < m_vecBitPermanenceActive[u] )
            {
                m_vecBitPermanenceMaximum[u] = m_vecBitPermanenceActive[u];
            }
        }
    }

    //ds add to history
    vecDescriptorsLEFT.push_back( p_matDescriptorLEFT );
    vecDescriptorsRIGHT.push_back( p_matDescriptorRIGHT );

    //ds compute world point
    const CPoint3DWORLD vecXYZWORLD( p_matTransformationLEFTtoWORLD*p_vecXYZLEFT );

    //ds update mean
    vecPointXYZMean = ( vecPointXYZMean+vecXYZWORLD )/2.0;




    //ds get a noisy descriptor
    //const CDescriptor vecDescriptorLEFTNoisy( _getDescriptorWithAddedNoise( p_matDescriptorLEFT ) );
    //vecDescriptorsLEFTNoisy.push_back( vecDescriptorLEFTNoisy );

    /*ds logging
    if( 100 == vecDescriptorsLEFT.size( ) )
    {
        //ds create logging matrix: rows -> bits, cols -> descriptor numbers
        Eigen::Matrix< bool, DESCRIPTOR_SIZE_BITS, 100 > matDescriptorEvolution;

        //ds fill matrix: columnswise per descriptor
        for( uint32_t u = 0; u < 100; ++u )
        {
            //ds buffer descriptor
            const Eigen::Matrix< bool, DESCRIPTOR_SIZE_BITS, 1 > cDescriptor( CWrapperOpenCV::getDescriptorVector< bool >( vecDescriptorsLEFT[u] ) );

            //ds fill column
            for( uint32_t v = 0; v < DESCRIPTOR_SIZE_BITS; ++v )
            {
                matDescriptorEvolution(v,u) = cDescriptor(v);
            }
        }

        //ds write stats to file
        std::ofstream ofLogfileEvolutionMap( "logs/bit_evolution_map_"+std::to_string( uID )+".txt", std::ofstream::out );
        std::ofstream ofLogfileEvolutionMeanVariance( "logs/bit_evolution_mean_variance_"+std::to_string( uID )+".txt", std::ofstream::out );
        std::ofstream ofLogfileEvolutionMeanVarianceSorted( "logs/bit_evolution_mean_variance_"+std::to_string( uID )+"_sorted.txt", std::ofstream::out );

        //ds mean vector
        Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecBitMean( Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >::Zero( ) );

        //ds loop over eigen matrix and dump the first 50 values
        for( int64_t u = 0; u < matDescriptorEvolution.rows( ); ++u )
        {
            for( int64_t v = 0; v < matDescriptorEvolution.cols( ); ++v )
            {
                ofLogfileEvolutionMap << matDescriptorEvolution( u, v ) << " ";
                vecBitMean[u] += matDescriptorEvolution( u, v );
            }
            ofLogfileEvolutionMap << "\n";

            //ds compute mean
            vecBitMean[u] /= matDescriptorEvolution.cols( );
        }

        //ds bit variance
        Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecBitVariance( Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >::Zero( ) );

        //ds loop over eigen matrix and dump the first 50 values
        for( int64_t u = 0; u < matDescriptorEvolution.rows( ); ++u )
        {
            for( int64_t v = 0; v < matDescriptorEvolution.cols( ); ++v )
            {
                vecBitVariance[u] += ( vecBitMean[u]-matDescriptorEvolution( u, v ) )*( vecBitMean[u]-matDescriptorEvolution( u, v ) );
            }

            //ds compute mean and plot it
            vecBitVariance[u] /= matDescriptorEvolution.cols( );

            ofLogfileEvolutionMeanVariance << u << " " << vecBitMean[u] << " " << vecBitVariance[u] << "\n";
        }

        //ds save files
        ofLogfileEvolutionMap.close( );
        ofLogfileEvolutionMeanVariance.close( );

        //ds get probabilities to vector for sorting
        std::vector< std::pair< uint32_t, double > > vecBitMeanSorted( DESCRIPTOR_SIZE_BITS );
        for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
        {
            vecBitMeanSorted[u] = std::make_pair( u, vecBitMean[u] );
        }

        //ds sort vector descending
        std::sort( vecBitMeanSorted.begin( ), vecBitMeanSorted.end( ), []( const std::pair< uint32_t, double > &prLHS, const std::pair< uint32_t, double > &pRHS ){ return prLHS.second > pRHS.second; } );

        //ds loop over sorted vector and dump the values parallel into columns
        for( const std::pair< uint32_t, double >& prBitMean: vecBitMeanSorted )
        {
            ofLogfileEvolutionMeanVarianceSorted << prBitMean.first << " " << prBitMean.second << " " << vecBitVariance[prBitMean.first] << "\n";
        }

        //ds close
        ofLogfileEvolutionMeanVarianceSorted.close( );
    }*/





        /*ds write stats to file
        std::ofstream ofLogfile( "logs/bit_mean_permanence_"+std::to_string( uID )+".txt", std::ofstream::out );

        //ds buffer values
        const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecBitProbabilities = getPDescriptorBRIEFLEFT( );
        const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecBitPermanence    = getBitPermanenceLEFT( );

        //ds get probabilities to vector for sorting
        std::vector< std::pair< uint32_t, double > > vecBitProbabilitiesSorted( DESCRIPTOR_SIZE_BITS );
        for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
        {
            vecBitProbabilitiesSorted[u] = std::make_pair( u, vecBitProbabilities[u] );
        }

        //ds sort vector descending
        std::sort( vecBitProbabilitiesSorted.begin( ), vecBitProbabilitiesSorted.end( ), []( const std::pair< uint32_t, double > &prLHS, const std::pair< uint32_t, double > &pRHS ){ return prLHS.second > pRHS.second; } );

        //ds loop over sorted vector and dump the values parallel into columns
        for( const std::pair< uint32_t, double >& prBitProbability: vecBitProbabilitiesSorted )
        {
            ofLogfile << prBitProbability.second << " " << vecBitPermanence[prBitProbability.first] << " " << prBitProbability.second+vecBitPermanence[prBitProbability.first] << "\n";
        }

        //ds save file
        ofLogfile.close( );*/
    //}

    //ds if acceptable
    //if( 129 > CWrapperOpenCV::getDistanceHammingProbability( vecPDescriptorLEFT, getPDescriptorBRIEFLEFT( ) ) )
    //{
        //ds add accumulated bit count
        m_vecSetBitsAccumulatedLEFT += vecDescriptorLEFT;
        ++m_uNumberOfBitsAccumulated;
    /*}
    else
    {
        std::cerr << uID << " " << CWrapperOpenCV::getDistanceHammingProbability( vecPDescriptorLEFT, getPDescriptorBRIEFLEFT( ) ) << std::endl;
    }*/

    //ds add the measurement to structure
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID,
                                                           p_ptUVLEFT,
                                                           p_ptUVRIGHT,
                                                           p_vecXYZLEFT,
                                                           vecXYZWORLD,
                                                           vecPointXYZOptimized,
                                                           p_matTransformationWORLDtoLEFT,
                                                           p_matProjectionWORLDtoLEFT,
                                                           p_matProjectionWORLDtoRIGHT,
                                                           uOptimizationsSuccessful ) );
}

void CLandmark::optimize( const UIDFrame& p_uFrame )
{
    //ds default false - gets set in optimization
    bIsOptimal = false;

    //ds update position - if we have at least n measurements
    if( CLandmark::uMinimumMeasurementsForOptimization < m_vecMeasurements.size( ) )
    {
        //vecPointXYZOptimized = _getOptimizedLandmarkLEFT3D( p_uFrame, vecPointXYZOptimized );
        vecPointXYZOptimized = _getOptimizedLandmarkSTEREOUV( p_uFrame, vecPointXYZOptimized );
    }
    else
    {
        bIsOptimal = true;
    }
}

//ds measurements reset
void CLandmark::clearMeasurements( const CPoint3DWORLD& p_vecXYZWORLD,
                                   const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                   const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                   const MatrixProjection& p_matProjectionWORLDtoRIGHT )
{
    //ds get a copy of the last measurement and modify it to the new conditions
    CMeasurementLandmark* pMeasurementLast = new CMeasurementLandmark( uID,
                                                                       m_vecMeasurements.back( )->ptUVLEFT,
                                                                       m_vecMeasurements.back( )->ptUVRIGHT,
                                                                       m_vecMeasurements.back( )->vecPointXYZLEFT,
                                                                       p_vecXYZWORLD,
                                                                       vecPointXYZOptimized,
                                                                       p_matTransformationWORLDtoLEFT,
                                                                       p_matProjectionWORLDtoLEFT,
                                                                       p_matProjectionWORLDtoRIGHT,
                                                                       uOptimizationsSuccessful );

    //ds free positions
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        delete pMeasurement;
    }

    //ds clear vector
    m_vecMeasurements.clear( );
    uOptimizationsFailed     = 0;
    uOptimizationsSuccessful = 0;

    //ds add last
    m_vecMeasurements.push_back( pMeasurementLast );
}

const uint64_t CLandmark::getSizeBytes( ) const
{
    //ds compute static size
    uint64_t uSizeBytes = sizeof( CLandmark );

    //ds add dynamic sizes: descriptor histories
    uSizeBytes += vecDescriptorsLEFT.size( )*sizeof( CDescriptor );
    uSizeBytes += vecDescriptorsRIGHT.size( )*sizeof( CDescriptor );

    //ds add dynamic sizes: measurements
    uSizeBytes += m_vecMeasurements.size( )*sizeof( CMeasurementLandmark );

    //ds done
    return uSizeBytes;
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkLEFT3D( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix3d matH( Eigen::Matrix3d::Zero( ) );
    Eigen::Vector3d vecB( Eigen::Vector3d::Zero( ) );
    const Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );
    double dErrorSquaredTotalMetersPREVIOUS = 0.0;

    //ds 3d point to optimize
    CPoint3DWORLD vecX( p_vecInitialGuess );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::uCapIterations; ++uIteration )
    {
        //ds counts
        double dErrorSquaredTotalMeters = 0.0;
        uint32_t uInliers               = 0;

        //ds initialize setup
        matH.setZero( );
        vecB.setZero( );

        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds get error
            const Eigen::Vector3d vecError( vecX-pMeasurement->vecPointXYZWORLD );

            //ds current error
            const double dErrorSquaredMeters = vecError.transpose( )*vecError;

            //ds check if outlier
            double dWeight = 1.0;
            if( 0.1 < dErrorSquaredMeters )
            {
                dWeight = 0.1/dErrorSquaredMeters;
            }
            else
            {
                ++uInliers;
            }
            dErrorSquaredTotalMeters += dWeight*dErrorSquaredMeters;

            //ds accumulate (special case as jacobian is the identity)
            matH += dWeight*matOmega;
            vecB += dWeight*vecError;
        }

        //ds update x solution
        vecX += matH.ldlt( ).solve( -vecB );

        //ds check if we have converged
        if( CLandmark::dConvergenceDelta > std::fabs( dErrorSquaredTotalMetersPREVIOUS-dErrorSquaredTotalMeters ) )
        {
            //ds compute average error
            const double dErrorSquaredAverageMeters = dErrorSquaredTotalMeters/m_vecMeasurements.size( );

            //ds if acceptable (don't mind about the actual number of inliers - we could be at this point with only 2 measurements -> 2 inliers -> 100%)
            if( 0 < uInliers )
            {
                //ds success
                ++uOptimizationsSuccessful;

                //ds update average
                dCurrentAverageSquaredError = dErrorSquaredAverageMeters;

                //ds check if optimal
                if( 0.075 > dErrorSquaredAverageMeters )
                {
                    bIsOptimal = true;
                }

                //std::printf( "<CLandmark>(_getOptimizedLandmarkWORLD) [%06lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f (inliers: %u/%lu)\n",
                //             uID, uOptimizationsSuccessful, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredErrorPixels, uInliers, m_vecMeasurements.size( ) );

                return vecX;
            }
            else
            {
                ++uOptimizationsFailed;
                //std::printf( "<CLandmark>(_getOptimizedLandmarkWORLD) landmark [%06lu] optimization failed - solution unacceptable (average error: %f, inliers: %u, iteration: %u)\n", uID, dErrorSquaredAverageMeters, uInliers, uIteration );

                //ds if still here the calibration did not converge - keep the initial estimate
                return p_vecInitialGuess;
            }
        }
        else
        {
            //ds update error
            dErrorSquaredTotalMetersPREVIOUS = dErrorSquaredTotalMeters;
        }
    }

    ++uOptimizationsFailed;
    //std::printf( "<CLandmark>(_getOptimizedLandmarkWORLD) landmark [%06lu] optimization failed - system did not converge\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkSTEREOUV( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( Eigen::Vector4d::Zero( ) );
    CPoint3DHomogenized vecX( CMiniVisionToolbox::getHomogeneous( p_vecInitialGuess ) );
    //Eigen::Matrix2d matOmega( Eigen::Matrix2d::Identity( ) );
    double dErrorSquaredTotalPixelsPREVIOUS = 0.0;

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::uCapIterations; ++uIteration )
    {
        //ds counts
        double dErrorSquaredTotalPixels = 0.0;
        uint32_t uInliers               = 0;

        //ds initialize setup
        matH.setZero( );
        vecB.setZero( );

        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds apply the projection to the transformed point
            const Eigen::Vector3d vecABCLEFT  = pMeasurement->matProjectionWORLDtoLEFT*vecX;
            const Eigen::Vector3d vecABCRIGHT = pMeasurement->matProjectionWORLDtoRIGHT*vecX;

            //ds buffer c value
            const double dCLEFT  = vecABCLEFT.z( );
            const double dCRIGHT = vecABCRIGHT.z( );

            //ds compute error
            const Eigen::Vector2d vecUVLEFT( vecABCLEFT.x( )/dCLEFT, vecABCLEFT.y( )/dCLEFT );
            const Eigen::Vector2d vecUVRIGHT( vecABCRIGHT.x( )/dCRIGHT, vecABCRIGHT.y( )/dCRIGHT );
            const Eigen::Vector4d vecError( vecUVLEFT.x( )-pMeasurement->ptUVLEFT.x,
                                            vecUVLEFT.y( )-pMeasurement->ptUVLEFT.y,
                                            vecUVRIGHT.x( )-pMeasurement->ptUVRIGHT.x,
                                            vecUVRIGHT.y( )-pMeasurement->ptUVRIGHT.y );

            //ds current error
            const double dErrorSquaredPixels = vecError.transpose( )*vecError;

            //std::printf( "[%06lu][%04u] error: %4.2f %4.2f %4.2f %4.2f (squared: %4.2f)\n", uID, uIteration, vecError(0), vecError(1), vecError(2), vecError(3) , dErrorSquaredPixels );

            //ds check if outlier
            double dWeight = 1.0;
            if( dKernelMaximumErrorSquaredPixels < dErrorSquaredPixels )
            {
                dWeight = dKernelMaximumErrorSquaredPixels/dErrorSquaredPixels;
            }
            else
            {
                ++uInliers;
            }
            dErrorSquaredTotalPixels += dWeight*dErrorSquaredPixels;

            //ds jacobian of the homogeneous division
            Eigen::Matrix< double, 2, 3 > matJacobianLEFT;
            matJacobianLEFT << 1/dCLEFT,          0, -vecABCLEFT.x( )/( dCLEFT*dCLEFT ),
                                      0,   1/dCLEFT, -vecABCLEFT.y( )/( dCLEFT*dCLEFT );

            Eigen::Matrix< double, 2, 3 > matJacobianRIGHT;
            matJacobianRIGHT << 1/dCRIGHT,           0, -vecABCRIGHT.x( )/( dCRIGHT*dCRIGHT ),
                                        0,   1/dCRIGHT, -vecABCRIGHT.y( )/( dCRIGHT*dCRIGHT );

            //ds final jacobian
            Eigen::Matrix< double, 4, 4 > matJacobian;
            matJacobian.setZero( );
            matJacobian.block< 2,4 >(0,0) = matJacobianLEFT*pMeasurement->matProjectionWORLDtoLEFT;
            matJacobian.block< 2,4 >(2,0) = matJacobianRIGHT*pMeasurement->matProjectionWORLDtoRIGHT;

            //ds precompute transposed
            const Eigen::Matrix< double, 4, 4 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds accumulate
            matH += dWeight*matJacobianTransposed*matJacobian;
            vecB += dWeight*matJacobianTransposed*vecError;
        }

        //ds solve constrained system (since dx(3) = 0.0) and update x solution
        vecX.block< 3,1 >(0,0) += matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB );

        //std::printf( "[%06lu][%04u]: %6.2f %6.2f %6.2f %6.2f (delta 2norm: %f inliers: %u)\n", uID, uIteration, vecX.x( ), vecX.y( ), vecX.z( ), vecX(3), vecDeltaX.squaredNorm( ), uInliers );

        //std::fprintf( m_pFilePosition, "%04lu %06lu %03u %03lu %03u %6.2f\n", p_uFrame, uID, uIteration, m_vecMeasurements.size( ), uInliers, dRSSCurrent );

        //ds check if we have converged
        if( CLandmark::dConvergenceDelta > std::fabs( dErrorSquaredTotalPixelsPREVIOUS-dErrorSquaredTotalPixels ) )
        {
            //ds compute average error
            const double dErrorSquaredAveragePixels = dErrorSquaredTotalPixels/m_vecMeasurements.size( );

            //ds if acceptable inlier/outlier ratio
            if( CLandmark::dMinimumRatioInliersToOutliers < static_cast< double >( uInliers )/m_vecMeasurements.size( ) )
            {
                //ds success
                ++uOptimizationsSuccessful;

                //ds update average
                dCurrentAverageSquaredError = dErrorSquaredAveragePixels;

                //ds check if optimal
                if( dMaximumErrorSquaredAveragePixels > dErrorSquaredAveragePixels )
                {
                    bIsOptimal = true;
                }

                //std::printf( "<CLandmark>(_getOptimizedLandmarkSTEREOUV) [%06lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f (inliers: %u/%lu)\n",
                //             uID, uOptimizationsSuccessful, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError, uInliers, m_vecMeasurements.size( ) );

                //ds update the estimate
                return vecX.block< 3,1 >(0,0);
            }
            else
            {
                ++uOptimizationsFailed;
                //std::printf( "<CLandmark>(_getOptimizedLandmarkSTEREOUV) landmark [%06lu] optimization failed - solution unacceptable (average error: %f, inliers: %u, iteration: %u)\n", uID, dErrorSquaredAveragePixels, uInliers, uIteration );

                //ds if still here the calibration did not converge - keep the initial estimate
                return p_vecInitialGuess;
            }
        }
        else
        {
            //ds update error
            dErrorSquaredTotalPixelsPREVIOUS = dErrorSquaredTotalPixels;
        }
    }

    ++uOptimizationsFailed;
    //std::printf( "<CLandmark>(_getOptimizedLandmarkSTEREOUV) landmark [%06lu] optimization failed - system did not converge\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkIDWA( )
{
    //ds return vector
    CPoint3DWORLD vecPointXYZWORLD( Eigen::Vector3d::Zero( ) );

    //ds total accumulated depth
    double dInverseDepthAccumulated = 0.0;

    //ds loop over all measurements
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        //ds current inverse depth
        const double dInverseDepth = 1.0/pMeasurement->vecPointXYZLEFT.z( );

        //ds add current measurement with depth weight
        vecPointXYZWORLD += dInverseDepth*pMeasurement->vecPointXYZWORLD;

        //std::cout << "in camera frame: " << pMeasurement->vecPointXYZ.transpose( ) << std::endl;
        //std::cout << "adding: " << dInverseDepth << " x " << pMeasurement->vecPointXYZWORLD.transpose( ) << std::endl;

        //ds accumulate depth
        dInverseDepthAccumulated += dInverseDepth;
    }

    //ds compute average point
    vecPointXYZWORLD /= dInverseDepthAccumulated;

    //std::cout << "from: " << vecPointXYZCalibrated.transpose( ) << " to: " << vecPointXYZWORLD.transpose( ) << std::endl;

    ++uOptimizationsSuccessful;

    double dSumSquaredErrors = 0.0;

    //ds loop over all previous measurements again
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        //ds get point into camera (cast to avoid eclipse error..)
        const CPoint3DHomogenized vecPointXYZLEFT( CMiniVisionToolbox::getHomogeneous( static_cast< CPoint3DCAMERA >( pMeasurement->matTransformationWORLDtoLEFT*vecPointXYZWORLD ) ) );

        //ds get projected point
        const CPoint2DHomogenized vecProjectionHomogeneous( m_matProjectionLEFT*vecPointXYZLEFT );

        //ds compute pixel coordinates TODO remove cast
        const Eigen::Vector2d vecUV = CWrapperOpenCV::getInterDistance( static_cast< Eigen::Vector2d >( vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) ), pMeasurement->ptUVLEFT );

        //ds compute squared error
        const double dSquaredError( vecUV.squaredNorm( ) );

        //ds add up
        dSumSquaredErrors += dSquaredError;
    }

    //ds average the measurement
    dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

    //ds if optimal
    if( 5.0 > dCurrentAverageSquaredError )
    {
        bIsOptimal = false;
    }

    //ds return
    return vecPointXYZWORLD;
}

const CDescriptor CLandmark::_getDescriptorWithAddedNoise( const CDescriptor& p_vecDescriptor ) const
{

#if defined NUMBER_OF_NOISY_BITS

    //ds set up random generator if necessary
    std::random_device cRandomDevice;
    std::mt19937 cGenerator( cRandomDevice( ) );
    std::uniform_int_distribution< > cDistributionBit( 0, DESCRIPTOR_SIZE_BITS-1 );
    std::uniform_int_distribution< > cDistributionValue( 0, 1 );

#endif

    //ds get it to bitset representation
    std::bitset< DESCRIPTOR_SIZE_BITS > vecDescriptor;

    //ds compute bytes (as  opencv descriptors are bytewise)
    const uint32_t uDescriptorSizeBytes = DESCRIPTOR_SIZE_BITS/8;

    //ds loop over all bytes
    for( uint32_t u = 0; u < uDescriptorSizeBytes; ++u )
    {
        //ds get minimal datafrom cv::mat
        const uchar chValue = p_vecDescriptor.at< uchar >( u );

        //ds get bitstring
        for( uint8_t v = 0; v < 8; ++v )
        {
            vecDescriptor[u*8+v] = ( chValue >> v ) & 1;
        }
    }

#if defined NUMBER_OF_NOISY_BITS

    //ds sample flip bits
    for( uint32_t u = 0; u < NUMBER_OF_NOISY_BITS; ++u )
    {
        vecDescriptor[cDistributionBit( cGenerator )] = cDistributionValue( cGenerator );
    }

#endif

    //ds new descriptor
    CDescriptor vecDescriptorOpenCV( p_vecDescriptor.clone( ) );

    //ds convert back to opencv - loop over all bytes
    for( uint32_t u = 0; u < uDescriptorSizeBytes; ++u )
    {
        //ds get minimal datafrom cv::mat
        uchar chValue = 0;

        //ds get bitstring
        for( uint8_t v = 0; v < 8; ++v )
        {
            chValue |= vecDescriptor[u*8+v] << (7 - v);
        }

        vecDescriptorOpenCV.at< uchar >( u ) = chValue;
    }

    return vecDescriptorOpenCV;

}
