#include "CKeyFrame.h"
#include "CBNode.h"



CKeyFrame::CKeyFrame( const std::vector< CKeyFrame* >::size_type& p_uID,
                      const uint64_t& p_uFrame,
                      const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
                      const CLinearAccelerationIMU& p_vecLinearAcceleration,
                      const std::vector< const CMeasurementLandmark* >& p_vecMeasurements,
                      const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud,
                      const uint32_t& p_uCountInstability,
                      const double& p_dMotionScaling,
                      const std::vector< const CMatchICP* > p_vecLoopClosures ): uID( p_uID ),
                                                                                uFrameOfCreation( p_uFrame ),
                                                                                matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                                                vecLinearAccelerationNormalized( p_vecLinearAcceleration ),
                                                                                vecMeasurements( p_vecMeasurements ),
                                                                                vecCloud( p_vecCloud ),
#if defined USING_BTREE and defined USING_BOW
                                                                                vecDescriptorPoolBTree( getDescriptorPoolBTree( vecCloud ) ),
                                                                                vecDescriptorPoolBoW( getDescriptorPoolBoW( vecCloud ) ),
#else
                                                                                vecDescriptorPool( getDescriptorPool( vecCloud ) ),
#endif
                                                                                uCountInstability( p_uCountInstability ),
                                                                                dMotionScaling( p_dMotionScaling ),
                                                                                vecLoopClosures( p_vecLoopClosures )
#if defined USING_BTREE and defined USING_BOW
                                                                                ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPoolBTree ) )
#elif defined USING_BTREE
                                                                                ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#elif defined USING_BF
                                                                                ,m_pMatcherBF( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) )
#elif defined USING_LSH
                                                                                ,m_pMatcherLSH( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) )
#endif
{
#if defined USING_BF
    m_pMatcherBF->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
#elif defined USING_LSH
    m_pMatcherLSH->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
    m_pMatcherLSH->train( );
#endif

    assert( !vecCloud->empty( ) );

    //ds save the cloud to a file
    //saveCloudToFile( );
}

CKeyFrame::CKeyFrame( const std::vector< CKeyFrame* >::size_type& p_uID,
                      const uint64_t& p_uFrame,
                      const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
                      const CLinearAccelerationIMU& p_vecLinearAcceleration,
                      const std::vector< const CMeasurementLandmark* >& p_vecMeasurements,
                      const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud,
                      const uint32_t& p_uCountInstability,
                      const double& p_dMotionScaling ): uID( p_uID ),
                                                        uFrameOfCreation( p_uFrame ),
                                                        matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                        vecLinearAccelerationNormalized( p_vecLinearAcceleration ),
                                                        vecMeasurements( p_vecMeasurements ),
                                                        vecCloud( p_vecCloud ),
#if defined USING_BTREE and defined USING_BOW
                                                        vecDescriptorPoolBTree( getDescriptorPoolBTree( vecCloud ) ),
                                                        vecDescriptorPoolBoW( getDescriptorPoolBoW( vecCloud ) ),
#else
                                                        vecDescriptorPool( getDescriptorPool( vecCloud ) ),
#endif
                                                        uCountInstability( p_uCountInstability ),
                                                        dMotionScaling( p_dMotionScaling )
#if defined USING_BTREE and defined USING_BOW
                                                        ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPoolBTree ) )
#elif defined USING_BTREE
                                                        ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#elif defined USING_BF
                                                        ,m_pMatcherBF( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) )
#elif defined USING_LSH
                                                        ,m_pMatcherLSH( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) )
#endif
{
#if defined USING_BF
    m_pMatcherBF->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
#elif defined USING_LSH
    m_pMatcherLSH->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
    m_pMatcherLSH->train( );
#endif

    assert( !vecCloud->empty( ) );

    //ds save the cloud to a file
    //saveCloudToFile( );
}

CKeyFrame::CKeyFrame( const std::string& p_strFile ): uID( std::stoi( p_strFile.substr( p_strFile.length( )-12, 6 ) ) ),
                                                      uFrameOfCreation( 0 ),
                                                      matTransformationLEFTtoWORLD( Eigen::Matrix4d::Identity( ) ),
                                                      vecLinearAccelerationNormalized( CLinearAccelerationIMU( 0.0, 0.0, 0.0 ) ),
                                                      vecMeasurements( std::vector< const CMeasurementLandmark* >( 0 ) ),
                                                      vecCloud( getCloudFromFile( p_strFile ) ),
#if defined USING_BTREE and defined USING_BOW
                                                      vecDescriptorPoolBTree( getDescriptorPoolBTree( vecCloud ) ),
                                                      vecDescriptorPoolBoW( getDescriptorPoolBoW( vecCloud ) ),
#else
                                                      vecDescriptorPool( getDescriptorPool( vecCloud ) ),
#endif
                                                      uCountInstability( 0 ),
                                                      dMotionScaling( 1.0 ),
                                                      vecLoopClosures( std::vector< const CMatchICP* >( 0 ) )
#if defined USING_BTREE and defined USING_BOW
                                                      ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPoolBTree ) )
#elif defined USING_BTREE
                                                      ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#elif defined USING_BF
                                                      ,m_pMatcherBF( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) )
#elif defined USING_LSH
                                                      ,m_pMatcherLSH( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) )
#endif
{
#if defined USING_BF
    m_pMatcherBF->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
#elif defined USING_LSH
    m_pMatcherLSH->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
    m_pMatcherLSH->train( );
#endif

    assert( 0 != vecCloud );
}

CKeyFrame::~CKeyFrame( )
{
    //ds free loop closures
    for( const CMatchICP* pClosure: vecLoopClosures )
    {
        if( 0 != pClosure )
        {
            delete pClosure;
        }

        for( CDescriptorVectorPoint3DWORLD* pPoint: *vecCloud )
        {
            delete pPoint;
        }
    }
}

void CKeyFrame::saveCloudToFile( ) const
{
    //ds construct filestring and open dump file
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "clouds/keyframe_%06lu.cloud", uID );
    std::ofstream ofCloud( chBuffer, std::ofstream::out );
    assert( ofCloud.is_open( ) );
    assert( ofCloud.good( ) );

    //ds dump pose and number of points information
    for( uint8_t u = 0; u < 4; ++u )
    {
        for( uint8_t v = 0; v < 4; ++v )
        {
            CLogger::writeDatum( ofCloud, matTransformationLEFTtoWORLD(u,v) );
        }
    }
    CLogger::writeDatum( ofCloud, vecCloud->size( ) );

    for( const CDescriptorVectorPoint3DWORLD* pPoint: *vecCloud )
    {
        //ds dump position and descriptor number info
        CLogger::writeDatum( ofCloud, pPoint->vecPointXYZWORLD.x( ) );
        CLogger::writeDatum( ofCloud, pPoint->vecPointXYZWORLD.y( ) );
        CLogger::writeDatum( ofCloud, pPoint->vecPointXYZWORLD.z( ) );
        CLogger::writeDatum( ofCloud, pPoint->vecPointXYZCAMERA.x( ) );
        CLogger::writeDatum( ofCloud, pPoint->vecPointXYZCAMERA.y( ) );
        CLogger::writeDatum( ofCloud, pPoint->vecPointXYZCAMERA.z( ) );

        assert( pPoint->ptUVLEFT.y == pPoint->ptUVRIGHT.y );

        CLogger::writeDatum( ofCloud, pPoint->ptUVLEFT.x );
        CLogger::writeDatum( ofCloud, pPoint->ptUVLEFT.y );
        CLogger::writeDatum( ofCloud, pPoint->ptUVRIGHT.x );
        CLogger::writeDatum( ofCloud, pPoint->ptUVRIGHT.y );

        CLogger::writeDatum( ofCloud, pPoint->vecDescriptors.size( ) );

        //ds dump all descriptors found so far
        for( const CDescriptor& pDescriptorLEFT: pPoint->vecDescriptors )
        {
            //ds print the descriptor elements
            for( int32_t u = 0; u < pDescriptorLEFT.cols; ++u ){ CLogger::writeDatum( ofCloud, pDescriptorLEFT.data[u] ); }
        }
    }

    ofCloud.close( );
}

std::shared_ptr< const std::vector< CMatchCloud > > CKeyFrame::getMatchesVisualSpatial( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloudQuery ) const
{
    std::shared_ptr< std::vector< CMatchCloud > > vecMatches( std::make_shared< std::vector< CMatchCloud > >( ) );

    /*ds treated points
    std::set< UIDLandmark > vecMatchedIDs;

    //ds try to match the query into the training
    for( const CDescriptorVectorPoint3DWORLD& cPointQuery: *p_vecCloudQuery )
    {
        //ds current matching distance
        double dMatchingDistanceBest = CKeyFrame::m_dCloudMatchingMatchingDistanceCutoff;
        std::vector< CDescriptorVectorPoint3DWORLD >::const_iterator itPointReferenceBest = vecCloud->end( );

        //ds match this point against all training remaining points
        for( std::vector< CDescriptorVectorPoint3DWORLD >::const_iterator itPointReference = vecCloud->begin( ); itPointReference != vecCloud->end( ); ++itPointReference )
        {
            //ds if not already added
            if( vecMatchedIDs.end( ) == vecMatchedIDs.find( itPointReference->uID ) )
            {
                //ds compute current matching distance against training point - EUCLIDIAN
                const double dMatchingDistanceEuclidian = CKeyFrame::m_dCloudMatchingWeightEuclidian*( cPointQuery.vecPointXYZCAMERA-itPointReference->vecPointXYZCAMERA ).squaredNorm( );

                //ds check if euclidian by itself is within search range
                if( dMatchingDistanceBest > dMatchingDistanceEuclidian )
                {
                    //ds get start and end iterators of training descriptors
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainStart( itPointReference->vecDescriptors.begin( ) );
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainEnd( itPointReference->vecDescriptors.end( )-1 );

                    double dMatchingDistanceHammingForward   = 0.0;
                    double dMatchingDistanceHammingBackwards = 0.0;
                    UIDDescriptor uDescriptorMatchings       = 0;

                    //ds compute descriptor matching in ascending versus descending order
                    for( const CDescriptor& cDescriptorQuery: cPointQuery.vecDescriptors )
                    {
                        dMatchingDistanceHammingForward   += cv::norm( cDescriptorQuery, *itDescriptorTrainStart, cv::NORM_HAMMING );
                        dMatchingDistanceHammingBackwards += cv::norm( cDescriptorQuery, *itDescriptorTrainEnd, cv::NORM_HAMMING );
                        ++uDescriptorMatchings;

                        //ds move forward respective backwards
                        ++itDescriptorTrainStart;

                        //ds check if descriptor arrays overlap
                        if( itPointReference->vecDescriptors.end( ) == itDescriptorTrainStart || itPointReference->vecDescriptors.begin( ) == itDescriptorTrainEnd )
                        {
                            break;
                        }

                        //ds since begin returns still a valid element whereas end does not
                        --itDescriptorTrainEnd;
                    }

                    //ds average matchings
                    dMatchingDistanceHammingForward   /= uDescriptorMatchings;
                    dMatchingDistanceHammingBackwards /= uDescriptorMatchings;

                    //std::printf( "matching forward: %f backwards: %f (matchings: %u)\n", dMatchingDistanceHammingForward, dMatchingDistanceHammingBackwards, uDescriptorMatchings );

                    //ds descriptor distance (take the better one)
                    const double dMatchingDistanceDescriptor = std::min( dMatchingDistanceHammingForward, dMatchingDistanceHammingBackwards );

                    //ds check if better (and in cutoff range)
                    if( dMatchingDistanceBest > dMatchingDistanceEuclidian+dMatchingDistanceDescriptor )
                    {
                        dMatchingDistanceBest = dMatchingDistanceEuclidian+dMatchingDistanceDescriptor;
                        itPointReferenceBest  = itPointReference;
                        //std::printf( "matching distance euclidian: %5.2f descriptor: %5.2f\n", dMatchingDistanceEuclidian, dMatchingDistanceDescriptor );
                    }
                }
            }
        }

        //ds check if we found a match
        if( itPointReferenceBest != vecCloud->end( ) )
        {
            //ds add the match
            vecMatches->push_back( CMatchCloud( cPointQuery, CDescriptorVectorPoint3DWORLD( *itPointReferenceBest ), dMatchingDistanceBest ) );
            vecMatchedIDs.insert( itPointReferenceBest->uID );
        }

        //std::printf( "(main) best match for query: %03lu -> train: %03lu (matching distance: %9.4f, descriptor matchings: %03lu)\n", cPointQuery.uID, uIDQueryBest, dMatchingDistanceBest, uDescriptorMatchingsBest );
    }*/

    return vecMatches;
}

std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > CKeyFrame::getCloudFromFile( const std::string& p_strFile )
{
    //ds open the file
    std::ifstream ifMessages( p_strFile, std::ifstream::in );

    //ds check if opening failed
    if( !ifMessages.is_open( ) )
    {
        throw std::invalid_argument( "invalid cloud file" );
    }

    //ds parse pose
    Eigen::Isometry3d matPose( Eigen::Matrix4d::Identity( ) );
    for( uint8_t u = 0; u < 4; ++u )
    {
        for( uint8_t v = 0; v < 4; ++v )
        {
            CLogger::readDatum( ifMessages, matPose(u,v) );
        }
    }

    //ds set pose
    matTransformationLEFTtoWORLD = matPose;

    //ds parse number of points
    std::vector< CLandmark* >::size_type uNumberOfPoints;
    CLogger::readDatum( ifMessages, uNumberOfPoints );

    //ds points in the cloud (preallocation ignored since const elements)
    std::shared_ptr< std::vector< CDescriptorVectorPoint3DWORLD* > > vecPoints( std::make_shared< std::vector< CDescriptorVectorPoint3DWORLD* > >( ) );

    //ds for all these points
    for( std::vector< CLandmark* >::size_type u = 0; u < uNumberOfPoints; ++u )
    {
        //ds point field
        CPoint3DWORLD vecPointXYZWORLD;
        CPoint3DCAMERA vecPointXYZCAMERA;
        CLogger::readDatum( ifMessages, vecPointXYZWORLD.x( ) );
        CLogger::readDatum( ifMessages, vecPointXYZWORLD.y( ) );
        CLogger::readDatum( ifMessages, vecPointXYZWORLD.z( ) );
        CLogger::readDatum( ifMessages, vecPointXYZCAMERA.x( ) );
        CLogger::readDatum( ifMessages, vecPointXYZCAMERA.y( ) );
        CLogger::readDatum( ifMessages, vecPointXYZCAMERA.z( ) );

        assert( 0.0 < vecPointXYZCAMERA.z( ) );

        cv::Point2d ptUVLEFT;
        cv::Point2d ptUVRIGHT;
        CLogger::readDatum( ifMessages, ptUVLEFT.x );
        CLogger::readDatum( ifMessages, ptUVLEFT.y );
        CLogger::readDatum( ifMessages, ptUVRIGHT.x );
        CLogger::readDatum( ifMessages, ptUVRIGHT.y );

        assert( ptUVLEFT.y == ptUVRIGHT.y );

        //ds number of descriptors
        std::vector< CMeasurementLandmark* >::size_type uNumberOfDescriptors;
        CLogger::readDatum( ifMessages, uNumberOfDescriptors );

        //ds descriptor vector (preallocate)
        std::vector< CDescriptor > vecDescriptors( uNumberOfDescriptors );

        //ds parse all descriptors
        for( std::vector< CMeasurementLandmark* >::size_type v = 0; v < uNumberOfDescriptors; ++v )
        {
            //ds current descriptor
            CDescriptor matDescriptor( 1, DESCRIPTOR_SIZE_BYTES, CV_8U );

            //ds every descriptor contains 64 fields
            for( uint32_t w = 0; w < DESCRIPTOR_SIZE_BYTES; ++w )
            {
                CLogger::readDatum( ifMessages, matDescriptor.data[w] );
            }

            vecDescriptors[v] = matDescriptor;
        }

        //ds set vector
        vecPoints->push_back( new CDescriptorVectorPoint3DWORLD( u, vecPointXYZWORLD, vecPointXYZCAMERA, ptUVLEFT, ptUVRIGHT, vecDescriptors ) );
    }

    return vecPoints;
}

const uint64_t CKeyFrame::getSizeBytes( ) const
{
    //ds compute static size
    uint64_t uSizeBytes = sizeof( CKeyFrame );

    //ds add dynamic sizes
    uSizeBytes += vecCloud->size( )*sizeof( CDescriptorVectorPoint3DWORLD );

    for( const CDescriptorVectorPoint3DWORLD* pPoint: *vecCloud )
    {
        uSizeBytes += pPoint->vecDescriptors.size( )*sizeof( CDescriptor );
    }

    uSizeBytes += vecLoopClosures.size( )*sizeof( CMatchICP );

    for( const CMatchICP* pMatch: vecLoopClosures )
    {
        uSizeBytes += pMatch->vecMatches->size( )*sizeof( CMatchCloud );
    }

    //uSizeBytes += vecDescriptorPool.size( )*sizeof( CDescriptorBRIEF< > );
    uSizeBytes += sizeof( CDescriptors );

    //ds done
    return uSizeBytes;
}
#if defined USING_BTREE and defined USING_BOW

const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > CKeyFrame::getDescriptorPoolBTree( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    mapDescriptorToPoint.clear( );
    std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPool;

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors
        for( const CDescriptor& cDescriptor: pPointWithDescriptors->vecDescriptors )
        {
            //ds map descriptor pool to points for later retrieval
            mapDescriptorToPoint.insert( std::make_pair( vecDescriptorPool.size( ), pPointWithDescriptors ) );
            vecDescriptorPool.push_back( CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >( vecDescriptorPool.size( ), CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >::getDescriptorVector( cDescriptor ) ) );
        }
    }

    return vecDescriptorPool;
}

const std::vector< boost::dynamic_bitset< > > CKeyFrame::getDescriptorPoolBoW( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    std::vector< boost::dynamic_bitset< > > vecDescriptorPool;

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors
        for( const CDescriptor& cDescriptor: pPointWithDescriptors->vecDescriptors )
        {
            //ds boost bitset
            boost::dynamic_bitset< > vecDescriptor( DESCRIPTOR_SIZE_BITS );

            //ds compute bytes (as  opencv descriptors are bytewise)
            const uint32_t uDescriptorSizeBytes = DESCRIPTOR_SIZE_BITS/8;

            //ds loop over all bytes
            for( uint32_t u = 0; u < uDescriptorSizeBytes; ++u )
            {
                //ds get minimal datafrom cv::mat
                const uchar chValue = cDescriptor.at< uchar >( u );

                //ds get bitstring
                for( uint8_t v = 0; v < 8; ++v )
                {
                    vecDescriptor[u*8+v] = ( chValue >> v ) & 1;
                }
            }

            vecDescriptorPool.push_back( vecDescriptor );
        }
    }

    return vecDescriptorPool;
}

#elif defined USING_BTREE

const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > CKeyFrame::getDescriptorPool( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    mapDescriptorToPoint.clear( );
    std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPool;

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors
        for( const CDescriptor& cDescriptor: pPointWithDescriptors->vecDescriptors )
        {
            //ds map descriptor pool to points for later retrieval
            mapDescriptorToPoint.insert( std::make_pair( vecDescriptorPool.size( ), pPointWithDescriptors ) );
            vecDescriptorPool.push_back( CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >( vecDescriptorPool.size( ), CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >::getDescriptorVector( cDescriptor ) ) );
        }
    }

    return vecDescriptorPool;
}

#elif defined USING_BTREE_INDEXED

const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > CKeyFrame::getDescriptorPool( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    mapDescriptorToPoint.clear( );
    std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPool;

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors
        for( const CDescriptor& cDescriptor: pPointWithDescriptors->vecDescriptors )
        {
            //ds map descriptor pool to points for later retrieval
            mapDescriptorToPoint.insert( std::make_pair( vecDescriptorPool.size( ), pPointWithDescriptors ) );
            vecDescriptorPool.push_back( CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >( vecDescriptorPool.size( ), CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >::getDescriptorVector( cDescriptor ), uID ) );
        }
    }

    return vecDescriptorPool;
}

#elif defined USING_BOW

const std::vector< boost::dynamic_bitset< > > CKeyFrame::getDescriptorPool( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    mapDescriptorToPoint.clear( );
    std::vector< boost::dynamic_bitset< > > vecDescriptorPool;

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors
        for( const CDescriptor& cDescriptor: pPointWithDescriptors->vecDescriptors )
        {
            //ds map descriptor pool to points for later retrieval
            mapDescriptorToPoint.insert( std::make_pair( vecDescriptorPool.size( ), pPointWithDescriptors ) );

            //ds boost bitset
            boost::dynamic_bitset< > vecDescriptor( DESCRIPTOR_SIZE_BITS );

            //ds compute bytes (as  opencv descriptors are bytewise)
            const uint32_t uDescriptorSizeBytes = DESCRIPTOR_SIZE_BITS/8;

            //ds loop over all bytes
            for( uint32_t u = 0; u < uDescriptorSizeBytes; ++u )
            {
                //ds get minimal datafrom cv::mat
                const uchar chValue = cDescriptor.at< uchar >( u );

                //ds get bitstring
                for( uint8_t v = 0; v < 8; ++v )
                {
                    vecDescriptor[u*8+v] = ( chValue >> v ) & 1;
                }
            }

            vecDescriptorPool.push_back( vecDescriptor );
        }
    }

    return vecDescriptorPool;
}

#else

const CDescriptors CKeyFrame::getDescriptorPool( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    mapDescriptorToPoint.clear( );
    CDescriptors vecDescriptorPool( 0, DESCRIPTOR_SIZE_BYTES, CV_8U );

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors row-wise
        for( const CDescriptor& cDescriptor: pPointWithDescriptors->vecDescriptors )
        {
            //ds consistency check
            //assert( pPointWithDescriptors->uID == mapDescriptorToPoint.at( vecDescriptorPool.rows )->uID );

            //ds map descriptor pool to points for later retrieval
            mapDescriptorToPoint.insert( std::make_pair( vecDescriptorPool.rows, pPointWithDescriptors ) );

            //ds increase pool
            vecDescriptorPool.push_back( cDescriptor );
        }
    }

    return vecDescriptorPool;
}
#endif
