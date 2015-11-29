#include "CTriangulator.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"

CTriangulator::CTriangulator( const std::shared_ptr< CStereoCamera > p_pStereoCamera,
                              const std::shared_ptr< cv::DescriptorExtractor > p_pExtractor,
                              const std::shared_ptr< cv::DescriptorMatcher > p_pMatcher,
                              const float& p_fMatchingDistanceCutoff ): m_pCameraSTEREO( p_pStereoCamera ),
                                                                          m_pCameraLEFT( p_pStereoCamera->m_pCameraLEFT ),
                                                                          m_pCameraRIGHT( p_pStereoCamera->m_pCameraRIGHT ),
                                                                                            m_pExtractor( p_pExtractor ),
                                                                                            m_pMatcher( p_pMatcher ),
                                                                                            m_fMatchingDistanceCutoff( p_fMatchingDistanceCutoff ),
                                                                                            m_uLimitedSearchRange( m_uLimitedSearchRangeToLEFT+m_uLimitedSearchRangeToRIGHT ),
                                                                                            m_uAdaptiveSteps( 10 )
{
    CLogger::openBox( );
    std::printf( "<CTriangulator>(CTriangulator) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CTriangulator>(CTriangulator) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CTriangulator>(CTriangulator) matching distance cutoff: %f\n", m_fMatchingDistanceCutoff );
    std::printf( "<CTriangulator>(CTriangulator) instance allocated\n" );
    CLogger::closeBox( );
}

CTriangulator::~CTriangulator( )
{
    std::printf( "<CTriangulator>(~CTriangulator) instance deallocated\n" );
}

const CMatchTriangulation CTriangulator::getPointTriangulatedInRIGHT( const cv::Mat& p_matImageRIGHT,
                                                       const float& p_fUTopLeft,
                                                       const float& p_fVTopLeft,
                                                       const float& p_fKeyPointSizePixels,
                                                       const cv::Point2f& p_ptUVLEFT,
                                                       const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds compute search range - overflow checking required in right
    const float fBorderCenter = 4*p_fKeyPointSizePixels;
    const float fFullHeight   = 8*p_fKeyPointSizePixels+1;
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::round( std::min( CTriangulator::fMinimumSearchRangePixels+fFullHeight, m_pCameraRIGHT->m_fWidthPixels-p_fUTopLeft ) );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToRIGHT = 0; uToRIGHT < uSearchRangeComplete; ++uToRIGHT )
    {
        vecPoolKeyPoints[uToRIGHT] = cv::KeyPoint( fBorderCenter+uToRIGHT, fBorderCenter, p_fKeyPointSizePixels );
    }

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT( cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete, fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( 0 == vecPoolKeyPoints.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( 0 == vecMatches.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) no match found" );
    }

    //ds current id
    const int32_t iIDMatch = vecMatches[0].trainIdx;

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( iIDMatch ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds buffer point
        const cv::Point2f ptUVRIGHT( vecPoolKeyPoints[iIDMatch].pt+cv::Point2f( p_fUTopLeft, p_fVTopLeft ) );

        //ds return triangulated point
        return CMatchTriangulation( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( p_ptUVLEFT,
                                                                                                ptUVRIGHT,
                                                                                                m_pCameraLEFT->m_matProjection,
                                                                                                m_pCameraRIGHT->m_matProjection ), ptUVRIGHT, matPoolDescriptors.row(iIDMatch) );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) matching distance" );
    }
}

const CMatchTriangulation CTriangulator::getPointTriangulatedInLEFT( const cv::Mat& p_matImageLEFT,
                                                       const float& p_fUTopLeft,
                                                       const float& p_fVTopLeft,
                                                       const float& p_fKeyPointSizePixels,
                                                       const cv::Point2f& p_ptUVRIGHT,
                                                       const CDescriptor& p_matReferenceDescriptorRIGHT ) const
{
    //ds compute search range - overflow checking required
    const float fBorderCenter = 4*p_fKeyPointSizePixels;
    const float fFullHeight   = 8*p_fKeyPointSizePixels+1;
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::round( std::min( CTriangulator::fMinimumSearchRangePixels+fFullHeight, m_pCameraLEFT->m_fWidthPixels-p_fUTopLeft ) );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToLEFT = 0; uToLEFT < uSearchRangeComplete; ++uToLEFT )
    {
        vecPoolKeyPoints[uToLEFT] = cv::KeyPoint( fBorderCenter+uToLEFT, fBorderCenter, p_fKeyPointSizePixels );
    }

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageLEFT( cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete, fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( 0 == vecPoolKeyPoints.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorRIGHT, matPoolDescriptors, vecMatches );

    if( 0 == vecMatches.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) no match found" );
    }

    //ds current id
    const int32_t iIDMatch = vecMatches[0].trainIdx;

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( iIDMatch ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds buffer point
        const cv::Point2f ptUVLEFT( vecPoolKeyPoints[iIDMatch].pt+cv::Point2f( p_fUTopLeft, p_fVTopLeft ) );

        //ds return triangulated point
        return CMatchTriangulation( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( ptUVLEFT,
                                                                                                p_ptUVRIGHT,
                                                                                                m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                                                m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection ), ptUVLEFT, matPoolDescriptors.row(iIDMatch) );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) matching distance" );
    }
}


const CMatchTriangulation CTriangulator::getPointTriangulatedInRIGHT( const cv::Mat& p_matImageRIGHT,
                                                       const float& p_fSearchRange,
                                                       const float& p_fUTopLeft,
                                                       const float& p_fVTopLeft,
                                                       const float& p_fKeyPointSizePixels,
                                                       const cv::Point2f& p_ptUVLEFT,
                                                       const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds compute search range - overflow checking required in right
    const float fBorderCenter = 4*p_fKeyPointSizePixels;
    const float fFullHeight   = 8*p_fKeyPointSizePixels+1;
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::round( std::min( p_fSearchRange+fFullHeight, m_pCameraRIGHT->m_fWidthPixels-p_fUTopLeft ) );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToRIGHT = 0; uToRIGHT < uSearchRangeComplete; ++uToRIGHT )
    {
        vecPoolKeyPoints[uToRIGHT] = cv::KeyPoint( fBorderCenter+uToRIGHT, fBorderCenter, p_fKeyPointSizePixels );
    }

    //cv::rectangle( p_matDisplayRIGHT, cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete, fFullHeight ), CColorCodeBGR( 255, 0, 0 ), 1 );

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT( cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete, fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( 0 == vecPoolKeyPoints.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( 0 == vecMatches.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) no match found" );
    }

    //ds current id
    const int32_t iIDMatch = vecMatches[0].trainIdx;

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( iIDMatch ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds buffer point
        const cv::Point2f ptUVRIGHT( vecPoolKeyPoints[iIDMatch].pt+cv::Point2f( p_fUTopLeft, p_fVTopLeft ) );

        //ds return triangulated point
        return CMatchTriangulation( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( p_ptUVLEFT,
                                                                                                ptUVRIGHT,
                                                                                                m_pCameraLEFT->m_matProjection,
                                                                                                m_pCameraRIGHT->m_matProjection ), ptUVRIGHT, matPoolDescriptors.row(iIDMatch) );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) matching distance" );
    }
}

const CMatchTriangulation CTriangulator::getPointTriangulatedInLEFT( const cv::Mat& p_matImageLEFT,
                                                       const float& p_fSearchRange,
                                                       const float& p_fUTopLeft,
                                                       const float& p_fVTopLeft,
                                                       const float& p_fKeyPointSizePixels,
                                                       const cv::Point2f& p_ptUVRIGHT,
                                                       const CDescriptor& p_matReferenceDescriptorRIGHT ) const
{
    //ds compute search range - overflow checking required
    const float fBorderCenter = 4*p_fKeyPointSizePixels;
    const float fFullHeight   = 8*p_fKeyPointSizePixels+1;
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::round( std::min( p_fSearchRange+fFullHeight, m_pCameraLEFT->m_fWidthPixels-p_fUTopLeft ) );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToLEFT = 0; uToLEFT < uSearchRangeComplete; ++uToLEFT )
    {
        vecPoolKeyPoints[uToLEFT] = cv::KeyPoint( fBorderCenter+uToLEFT, fBorderCenter, p_fKeyPointSizePixels );
    }

    //cv::rectangle( p_matDisplayLEFT, cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete, fFullHeight ), CColorCodeBGR( 255, 0, 0 ), 1 );

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageLEFT( cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete, fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( 0 == vecPoolKeyPoints.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorRIGHT, matPoolDescriptors, vecMatches );

    if( 0 == vecMatches.size( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) no match found" );
    }

    //ds current id
    const int32_t iIDMatch = vecMatches[0].trainIdx;

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( iIDMatch ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds buffer point
        const cv::Point2f ptUVLEFT( vecPoolKeyPoints[iIDMatch].pt+cv::Point2f( p_fUTopLeft, p_fVTopLeft ) );

        //ds return triangulated point
        return CMatchTriangulation( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( ptUVLEFT,
                                                                                                p_ptUVRIGHT,
                                                                                                m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                                                m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection ), ptUVLEFT, matPoolDescriptors.row(iIDMatch) );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) matching distance" );
    }
}

const CPoint3DCAMERA CTriangulator::getPointTriangulatedLimitedSVDLS( cv::Mat& p_matDisplayRIGHT, const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t& iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t& iVReference( p_cKeyPointLEFT.pt.y );
    const double& dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( std::max( iUReference-m_uLimitedSearchRangeToLEFT, static_cast< uint32_t >( 0 ) ) );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
        cv::circle( p_matDisplayRIGHT, vecPoolKeyPoints[u].pt, 1, CColorCodeBGR( 125, 125, 125 ), -1 );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( p_cKeyPointLEFT.pt,
                                                                           vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                           m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                           m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}

const CPoint3DCAMERA CTriangulator::getPointTriangulatedLimitedQRLS( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t& iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t& iVReference( p_cKeyPointLEFT.pt.y );
    const double& dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( iUReference-m_uLimitedSearchRangeToLEFT );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds precheck for nans
        const CPoint3DCAMERA vecTriangulatedPoint( CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( p_cKeyPointLEFT.pt,
                                                                                                                     vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                                                                     m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                                                                     m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection ) );

        //ds check if valid
        if( 0 != vecTriangulatedPoint.squaredNorm( ) )
        {
            return vecTriangulatedPoint;
        }
        else
        {
            throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) QRLS failed" );
        }
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}

const CPoint3DCAMERA CTriangulator::getPointTriangulatedLimitedSVDDLT( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t& iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t& iVReference( p_cKeyPointLEFT.pt.y );
    const double& dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( iUReference-m_uLimitedSearchRangeToLEFT );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDDLT( p_cKeyPointLEFT.pt,
                                                                            vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                            m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                            m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}
