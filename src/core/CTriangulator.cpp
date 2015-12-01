#include "CTriangulator.h"

#include "exceptions/CExceptionNoMatchFound.h"
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
                                                                                            m_dF( m_pCameraLEFT->m_matProjection(0,0) ),
                                                                                            m_dFInverse( 1/m_dF ),
                                                                                            m_dPu( m_pCameraLEFT->m_matProjection(0,2) ),
                                                                                            m_dPv( m_pCameraLEFT->m_matProjection(1,2) ),
                                                                                            m_dDuR( m_pCameraRIGHT->m_matProjection(0,3) ),
                                                                                            m_dDuRFlipped( -m_dDuR )
{
    //ds validate epipolar, rectified projection matrices
    assert( m_pCameraLEFT->m_matProjection(0,0) == m_pCameraRIGHT->m_matProjection(0,0) );
    assert( m_pCameraLEFT->m_matProjection(1,1) == m_pCameraRIGHT->m_matProjection(1,1) );
    assert( m_pCameraLEFT->m_matProjection(0,0) == m_pCameraRIGHT->m_matProjection(1,1) );
    assert( m_pCameraLEFT->m_matProjection(0,2) == m_pCameraRIGHT->m_matProjection(0,2) );
    assert( m_pCameraLEFT->m_matProjection(1,2) == m_pCameraRIGHT->m_matProjection(1,2) );
    assert( m_pCameraLEFT->m_matProjection(0,3) == 0.0 );
    assert( m_pCameraLEFT->m_matProjection(1,3) == 0.0 );
    assert( m_pCameraRIGHT->m_matProjection(1,3) == 0.0 );

    CLogger::openBox( );
    std::printf( "<CTriangulator>(CTriangulator) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CTriangulator>(CTriangulator) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CTriangulator>(CTriangulator) matching distance cutoff: %f\n", m_fMatchingDistanceCutoff );
    std::printf( "<CTriangulator>(CTriangulator) dF: %f\n", m_dF );
    std::printf( "<CTriangulator>(CTriangulator) dPu: %f\n", m_dPu );
    std::printf( "<CTriangulator>(CTriangulator) dPv: %f\n", m_dPv );
    std::printf( "<CTriangulator>(CTriangulator) DuR: %f\n", m_dDuR );
    std::printf( "<CTriangulator>(CTriangulator) minimum depth: %fm\n", m_dDuRFlipped/m_pCameraSTEREO->m_uPixelWidth );
    std::printf( "<CTriangulator>(CTriangulator) instance allocated\n" );
    CLogger::closeBox( );
}

CTriangulator::~CTriangulator( )
{
    std::printf( "<CTriangulator>(~CTriangulator) instance deallocated\n" );
}

const CMatchTriangulation CTriangulator::getPointTriangulatedInRIGHTFull( cv::Mat& p_matDisplaySTEREO, const cv::Mat& p_matImageRIGHT,
                                                       const float& p_fUTopLeft,
                                                       const float& p_fVTopLeft,
                                                       const float& p_fKeyPointSizePixels,
                                                       const cv::Point2f& p_ptUVLEFT,
                                                       const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds compute search range - overflow checking required in right
    const float fBorderCenter = 4*p_fKeyPointSizePixels;
    const float fFullHeight   = 8*p_fKeyPointSizePixels+1;

    //ds check search range
    if( p_ptUVLEFT.x <= p_fUTopLeft+fBorderCenter )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) insufficient search range" );
    }
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::ceil( p_ptUVLEFT.x-p_fUTopLeft-fBorderCenter );
    assert( 0 < uSearchRangeComplete );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToRIGHT = 0; uToRIGHT < uSearchRangeComplete; ++uToRIGHT )
    {
        vecPoolKeyPoints[uToRIGHT] = cv::KeyPoint( fBorderCenter+uToRIGHT, fBorderCenter, p_fKeyPointSizePixels );
    }

    cv::rectangle( p_matDisplaySTEREO, cv::Rect( m_pCameraLEFT->m_fWidthPixels+p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete+fFullHeight, fFullHeight ), CColorCodeBGR( 255, 0, 0 ), 1 );

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT( cv::Rect( p_fUTopLeft, p_fVTopLeft, std::min( uSearchRangeComplete+fFullHeight, m_pCameraRIGHT->m_fWidthPixels-p_fUTopLeft ), fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

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
        return CMatchTriangulation( getPointInLEFT( p_ptUVLEFT, ptUVRIGHT ), ptUVRIGHT, matPoolDescriptors.row(iIDMatch) );
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
    assert( false );

    //ds compute search range - overflow checking required
    const float fBorderCenter = 4*p_fKeyPointSizePixels;
    const float fFullHeight   = 8*p_fKeyPointSizePixels+1;
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::round( std::min( CTriangulator::fMinimumSearchRangePixels+fFullHeight, m_pCameraLEFT->m_fWidthPixels-p_fUTopLeft ) )-1;

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
        return CMatchTriangulation( getPointInLEFT( ptUVLEFT, p_ptUVRIGHT ), ptUVLEFT, matPoolDescriptors.row(iIDMatch) );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) matching distance" );
    }
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

    //ds check search range
    if( p_ptUVLEFT.x <= p_fUTopLeft+fBorderCenter )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInRIGHT) insufficient search range" );
    }
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::ceil( p_ptUVLEFT.x-p_fUTopLeft-fBorderCenter );
    assert( 0 < uSearchRangeComplete );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToRIGHT = 0; uToRIGHT < uSearchRangeComplete; ++uToRIGHT )
    {
        vecPoolKeyPoints[uToRIGHT] = cv::KeyPoint( fBorderCenter+uToRIGHT, fBorderCenter, p_fKeyPointSizePixels );
    }

    //cv::rectangle( p_matDisplayRIGHT, cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete+fFullHeight, fFullHeight ), CColorCodeBGR( 255, 0, 0 ), 1 );

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT( cv::Rect( p_fUTopLeft, p_fVTopLeft, std::min( uSearchRangeComplete+fFullHeight, m_pCameraRIGHT->m_fWidthPixels-p_fUTopLeft ), fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

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
        return CMatchTriangulation( getPointInLEFT( p_ptUVLEFT, ptUVRIGHT ), ptUVRIGHT, matPoolDescriptors.row(iIDMatch) );
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

    //ds check search range
    if( 0 >= p_fSearchRange )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) insufficient search range" );
    }
    const std::vector< cv::KeyPoint >::size_type uSearchRangeComplete = std::ceil( std::min( p_fSearchRange, m_pCameraLEFT->m_fWidthPixels-p_fUTopLeft ) )+1;
    assert( 0 < uSearchRangeComplete );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uSearchRangeComplete );

    //ds set the keypoints
    for( std::vector< cv::KeyPoint >::size_type uToLEFT = 0; uToLEFT < uSearchRangeComplete; ++uToLEFT )
    {
        vecPoolKeyPoints[uToLEFT] = cv::KeyPoint( fBorderCenter+uToLEFT+1, fBorderCenter, p_fKeyPointSizePixels );
    }

    //cv::rectangle( p_matDisplayLEFT, cv::Rect( p_fUTopLeft, p_fVTopLeft, uSearchRangeComplete+fFullHeight, fFullHeight ), CColorCodeBGR( 255, 0, 0 ), 1 );

    //ds compute descriptors
    CDescriptors matPoolDescriptors;
    m_pExtractor->compute( p_matImageLEFT( cv::Rect( p_fUTopLeft, p_fVTopLeft, std::min( uSearchRangeComplete+fFullHeight, m_pCameraLEFT->m_fWidthPixels-p_fUTopLeft ), fFullHeight ) ), vecPoolKeyPoints, matPoolDescriptors );

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
        return CMatchTriangulation( getPointInLEFT( ptUVLEFT, p_ptUVRIGHT ), ptUVLEFT, matPoolDescriptors.row(iIDMatch) );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedInLEFT) matching distance" );
    }
}

const CPoint3DCAMERA CTriangulator::getPointInLEFT( const cv::Point2f& p_ptUVLEFT, const cv::Point2f& p_ptUVRIGHT ) const
{
    //ds input validation
    assert( p_ptUVRIGHT.x < p_ptUVLEFT.x );
    assert( p_ptUVRIGHT.y == p_ptUVLEFT.y );

    //ds first compute depth (z in camera)
    const double dZ = m_dDuRFlipped/( p_ptUVLEFT.x-p_ptUVRIGHT.x );

    //ds set 3d point
    const CPoint3DCAMERA vecPointLEFT( m_dFInverse*dZ*( p_ptUVLEFT.x-m_dPu ),
                                       m_dFInverse*dZ*( p_ptUVLEFT.y-m_dPv ),
                                       dZ );

    //ds output validation (machine precision)
    assert( std::fabs( vecPointLEFT.x( )-m_dFInverse*( dZ*p_ptUVRIGHT.x-dZ*m_dPu-m_dDuR ) ) < 1e-10 );
    assert( std::fabs( vecPointLEFT.y( )-m_dFInverse*dZ*( p_ptUVRIGHT.y-m_dPv ) ) < 1e-10 );

    //ds return triangulated point
    return vecPointLEFT;
}
