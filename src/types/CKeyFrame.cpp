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
                                                                                //,vecBitMask( getBitMaskFiltered( vecCloud ) )
                                                                                ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#elif defined USING_BF
                                                                                ,m_pMatcherBF( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) )
#elif defined USING_LSH
                                                                                ,m_pMatcherLSH( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) )
#elif defined USING_BPTREE
                                                                                ,m_pBPTree( std::make_shared< CBPTree< MAXIMUM_DISTANCE_HAMMING_PROBABILITY, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
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
                                                        //,vecBitMask( getBitMaskFiltered( vecCloud ) )
                                                        ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#elif defined USING_BF
                                                        ,m_pMatcherBF( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) )
#elif defined USING_LSH
                                                        ,m_pMatcherLSH( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) )
#elif defined USING_BPTREE
                                                        ,m_pBPTree( std::make_shared< CBPTree< MAXIMUM_DISTANCE_HAMMING_PROBABILITY, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#endif
{

#if defined USING_BF

    m_pMatcherBF->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );

#elif defined USING_LSH

    m_pMatcherLSH->add( std::vector< CDescriptors >( 1, vecDescriptorPool ) );
    m_pMatcherLSH->train( );

#endif

    assert( !vecCloud->empty( ) );

    /*ds logging
    if( 4 == uID || 54 == uID || 36 == uID )
    {
        //ds create logging matrix: rows -> bits, cols -> points
        Eigen::MatrixXd matProbabilities( DESCRIPTOR_SIZE_BITS, vecCloud->size( ) );

        //ds fill the matrix
        for( uint32_t u = 0; u < vecCloud->size( ); ++u )
        {
            //ds buffer probabilities
            const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecPDescriptor( vecCloud->at( u )->vecPDescriptorBRIEF );

            //ds fill column
            for( uint32_t v = 0; v < DESCRIPTOR_SIZE_BITS; ++v )
            {
                matProbabilities(v,u) = vecPDescriptor(v);
            }
        }

        //ds write stats to file
        std::ofstream ofLogfile( "logs/bit_probability_map_"+std::to_string( uID )+"_"+std::to_string( DESCRIPTOR_SIZE_BITS )+"x"+std::to_string( vecCloud->size( ) )+".txt", std::ofstream::out );

        //ds loop over eigen matrix and dump the values
        for( int64_t u = 0; u < matProbabilities.rows( ); ++u )
        {
            for( int64_t v = 0; v < matProbabilities.cols( ); ++v )
            {
                ofLogfile << matProbabilities( u, v ) << " ";
            }

            ofLogfile << "\n";
        }

        //ds save file
        ofLogfile.close( );
    }*/

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
                                                      //,vecBitMask( getBitMaskFiltered( vecCloud ) )
                                                      ,m_pBTree( std::make_shared< CBTree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
#elif defined USING_BF
                                                      ,m_pMatcherBF( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) )
#elif defined USING_LSH
                                                      ,m_pMatcherLSH( std::make_shared< cv::FlannBasedMatcher >( new cv::flann::LshIndexParams( 1, 20, 2 ) ) )
#elif defined USING_BPTREE
                                                      ,m_pBPTree( std::make_shared< CBPTree< MAXIMUM_DISTANCE_HAMMING_PROBABILITY, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > >( uID, vecDescriptorPool ) )
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

        //ds set vector TODO update implementation
        //vecPoints->push_back( new CDescriptorVectorPoint3DWORLD( u, vecPointXYZWORLD, vecPointXYZCAMERA, ptUVLEFT, ptUVRIGHT, vecDescriptors ) );
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
            vecDescriptorPool.push_back( CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >( vecDescriptorPool.size( ), CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >::getDescriptorVector( cDescriptor ), uID, pPointWithDescriptors->uID ) );
        }
    }

    return vecDescriptorPool;
}

#elif defined USING_BITREE

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

#if defined SHUFFLE_DESCRIPTORS

    //ds shuffle complete vector
    std::random_shuffle( vecDescriptorPool.begin( ), vecDescriptorPool.end( ) );

#endif

    //ds return const
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

#elif defined USING_BPTREE or defined USING_BPITREE

const std::vector< CPDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > CKeyFrame::getDescriptorPool( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    //ds must have same size as measurements (=points)
    std::vector< CPDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > vecDescriptorPoolPBRIEF;
    vecDescriptorPoolPBRIEF.reserve( p_vecCloud->size( ) );

    //ds fill the pool - looping with an index as we need the position in the cloud for later retrieval
    for( uint64_t uIDPointInCloud = 0; uIDPointInCloud < p_vecCloud->size( ); ++uIDPointInCloud )
    {
        //ds map descriptor pool to points for later retrieval
        mapDescriptorToPoint.insert( std::make_pair( uIDPointInCloud, p_vecCloud->at( uIDPointInCloud ) ) );

        //ds add to pool
        vecDescriptorPoolPBRIEF.push_back( CPDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >( uIDPointInCloud, p_vecCloud->at( uIDPointInCloud )->vecPDescriptorBRIEF, uID ) );
    }

    return vecDescriptorPoolPBRIEF;
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

#if defined USING_BTREE or defined USING_BITREE

const std::bitset< DESCRIPTOR_SIZE_BITS > CKeyFrame::getBitMaskFiltered( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    //ds noisy bit counting
    Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecNoisyBitCounts( Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >::Zero( ) );

    //ds loop over all points
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds get descriptor probabilities
        const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >& vecProbabilities( pPointWithDescriptors->cBitStatisticsLEFT.vecBitProbabilities );

        //ds check if we have any noisy bits
        for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
        {
            //ds if the probability is around 50/50
            if( 0.1 < vecProbabilities[u] && 0.9 > vecProbabilities[u] )
            {
                //ds register noisy bit position
                vecNoisyBitCounts[u]=vecNoisyBitCounts[u]+1.0;
            }
        }
    }

    //ds get relative noise counts
    const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecNoisyBitCountsRelative( vecNoisyBitCounts/p_vecCloud->size( ) );

    /*ds write stats to file
    std::ofstream ofLogfile( "logs/noisy_bit_counts_"+std::to_string( uID )+".txt", std::ofstream::out );

    //ds loop over the set
    for( int32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
    {
        ofLogfile << u << " " << vecNoisyBitCountsRelative[u] << " " << vecNoisyBitCounts[u] << " " << p_vecCloud->size( ) << "\n";
    }

    //ds save file
    ofLogfile.close( );*/

    //ds allocate bitset and set all bits to available
    std::bitset< DESCRIPTOR_SIZE_BITS > vecBitMask;
    vecBitMask.set( );
    assert( DESCRIPTOR_SIZE_BITS == vecBitMask.count( ) );

    //ds disable selected bits
    for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
    {
        //ds if above threshold
        if( 0.25 < vecNoisyBitCountsRelative[u] )
        {
            //ds label this bit noisy and disable it
            vecBitMask[u] = false;
        }
    }

    //ds exit
    return vecBitMask;
}

std::vector< uint32_t > CKeyFrame::getSplitOrder( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud )
{
    //ds confident bits counting
    Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecAccumulatedProbabilities( Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 >::Zero( ) );

    //ds loop over all points
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        vecAccumulatedProbabilities += pPointWithDescriptors->cBitStatisticsLEFT.vecBitProbabilities;
    }

    //ds get average
    const Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > vecMean( vecAccumulatedProbabilities/p_vecCloud->size( ) );
    std::vector< std::pair< uint32_t, double > > vecVariance( DESCRIPTOR_SIZE_BITS );

    //ds compute variance
    for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
    {
        //ds buffer
        double dVariance = 0.0;

        for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
        {
            const double dDelta = pPointWithDescriptors->cBitStatisticsLEFT.vecBitProbabilities[u]-vecMean[u];
            dVariance += dDelta*dDelta;
        }

        //ds compute variance
        dVariance /= p_vecCloud->size( );

        //ds add to structure
        vecVariance[u] = std::make_pair( u, dVariance );
    }

    //ds sort vector descending
    std::sort( vecVariance.begin( ), vecVariance.end( ), []( const std::pair< uint32_t, double > &prLHS, const std::pair< uint32_t, double > &pRHS ){ return prLHS.second > pRHS.second; } );

    /*ds write stats to file
    std::ofstream ofLogfileBitConfidence( "logs/bit_confidence_"+std::to_string( uID )+".txt", std::ofstream::out );

    //ds loop over the set
    for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
    {
        ofLogfileBitConfidence << u << " " << vecVariance[u].second << "\n";
    }

    //ds save file
    ofLogfileBitConfidence.close( );*/

    //ds compute split vector
    std::vector< uint32_t > vecSplitOrder( DESCRIPTOR_SIZE_BITS );
    for( uint32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u )
    {
        vecSplitOrder[u] = vecVariance[u].first;
    }

    return vecSplitOrder;
}

//ds TODO REFACTOR
const std::vector< Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > > CKeyFrame::getBitProbabilities( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud ) const
{
    //ds result vector
    std::vector< Eigen::Matrix< double, DESCRIPTOR_SIZE_BITS, 1 > > vecBitProbabilities;

    //ds fill the pool
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        //ds add up descriptors
        for( uint64_t u = 0; u < pPointWithDescriptors->vecDescriptors.size( ); ++u )
        {
            vecBitProbabilities.push_back( pPointWithDescriptors->cBitStatisticsLEFT.vecBitProbabilities );
        }
    }

    return vecBitProbabilities;
}

const std::map< UIDLandmark, CBitStatistics > CKeyFrame::getBitStatistics( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD* > > p_vecCloud ) const
{
    //ds result map
    std::map< UIDLandmark, CBitStatistics > mapBitStatistics;

    //ds fill the map
    for( const CDescriptorVectorPoint3DWORLD* pPointWithDescriptors: *p_vecCloud )
    {
        mapBitStatistics.insert( std::make_pair( pPointWithDescriptors->uID, pPointWithDescriptors->cBitStatisticsLEFT ) );
    }

    assert( mapBitStatistics.size( ) == p_vecCloud->size( ) );
    return mapBitStatistics;
}

#endif
