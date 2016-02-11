#ifndef CPINHOLECAMERA_H
#define CPINHOLECAMERA_H

#include <iostream>

#include "types/Types.h"
#include "utility/CLogger.h"



class CPinholeCamera
{

public:

    //ds construct from parameter vector
    CPinholeCamera( const std::string& p_strLabel,
                    const uint32_t& p_uWidthPixels,
                    const uint32_t& p_uHeightPixels,
                    const MatrixProjection& p_matProjection,
                    const Eigen::Matrix3d& p_matIntrinsic,
                    const double& p_dFocalLengthMeters,
                    const Eigen::Vector4d& p_vecDistortionCoefficients,
                    const Eigen::Matrix3d& p_matRectification ): m_strCameraLabel( p_strLabel ),
                                                                               m_uWidthPixel( p_uWidthPixels ),
                                                                               m_uHeightPixel( p_uHeightPixels ),
                                                                                m_matProjection( p_matProjection ),
                                                                                m_matIntrinsic( p_matIntrinsic ),
                                                                                m_matIntrinsicP( m_matProjection.block< 3, 3 >( 0, 0 ) ),
                                                                                m_matIntrinsicInverse( m_matIntrinsic.inverse( ) ),
                                                                                m_matIntrinsicPInverse( m_matIntrinsicP.inverse( ) ),
                                                                                m_matIntrinsicInverseTransposed( m_matIntrinsicInverse.transpose( ) ),
                                                                                m_matIntrinsicPInverseTransposed( m_matIntrinsicPInverse.transpose( ) ),
                                                                                m_matIntrinsicTransposed( m_matIntrinsic.transpose( ) ),
                                                                                m_matIntrinsicPTransposed( m_matIntrinsicP.transpose( ) ),
                                                                                m_dFx( m_matIntrinsic(0,0) ),
                                                                                m_dFy( m_matIntrinsic(1,1) ),
                                                                                m_dFxP( m_matIntrinsicP(0,0) ),
                                                                                m_dFyP( m_matIntrinsicP(1,1) ),
                                                                                m_dFxNormalized( m_dFx/m_uWidthPixel ),
                                                                                m_dFyNormalized( m_dFy/m_uHeightPixel ),
                                                                                m_dCx( m_matIntrinsic(0,2) ),
                                                                                m_dCy( m_matIntrinsic(1,2) ),
                                                                                m_dCxP( m_matIntrinsicP(0,2) ),
                                                                                m_dCyP( m_matIntrinsicP(1,2) ),
                                                                                m_dCxNormalized( m_dCx/m_uWidthPixel ),
                                                                                m_dCyNormalized( m_dCy/m_uHeightPixel ),
                                                                                m_dFocalLengthMeters( p_dFocalLengthMeters ),
                                                                                m_vecDistortionCoefficients( p_vecDistortionCoefficients ),
                                                                                m_matRectification( p_matRectification ),
                                                                                m_vecPrincipalPoint( Eigen::Vector2d( m_dCx, m_dCy ) ),
                                                                                m_vecPrincipalPointNormalized( Eigen::Vector3d( m_dCxNormalized, m_dCyNormalized, 1.0 ) ),
                                                                                m_iWidthPixel( m_uWidthPixel ),
                                                                                m_iHeightPixel( m_uHeightPixel ),
                                                                                m_dWidthPixels( m_uWidthPixel ),
                                                                                m_dHeightPixels( m_uHeightPixel ),
                                                                                m_fWidthPixels( m_uWidthPixel ),
                                                                                m_fHeightPixels( m_uHeightPixel ),
                                                                                m_prRangeWidthNormalized( std::pair< double, double >( getNormalizedX( 0 ), getNormalizedX( m_uWidthPixel ) ) ),
                                                                                m_prRangeHeightNormalized( std::pair< double, double >( getNormalizedY( 0 ), getNormalizedY( m_uHeightPixel ) ) ),
                                                                                m_cFieldOfView( 28, 28, m_uWidthPixel-56, m_uHeightPixel-56 )
    {
        //ds log complete configuration
        _logConfiguration( 0.0 );
    }

    //ds no manual dynamic allocation
    virtual ~CPinholeCamera( ){ }

public:

    const std::string m_strCameraLabel;

    //ds intrinsics
    const uint32_t m_uWidthPixel;
    const uint32_t m_uHeightPixel;
    const MatrixProjection m_matProjection;
    const Eigen::Matrix3d m_matIntrinsic;
    const Eigen::Matrix3d m_matIntrinsicP;
    const Eigen::Matrix3d m_matIntrinsicInverse;
    const Eigen::Matrix3d m_matIntrinsicPInverse;
    const Eigen::Matrix3d m_matIntrinsicInverseTransposed;
    const Eigen::Matrix3d m_matIntrinsicPInverseTransposed;
    const Eigen::Matrix3d m_matIntrinsicTransposed;
    const Eigen::Matrix3d m_matIntrinsicPTransposed;
    const double m_dFx;
    const double m_dFy;
    const double m_dFxP;
    const double m_dFyP;
    const double m_dFxNormalized;
    const double m_dFyNormalized;
    const double m_dCx;
    const double m_dCy;
    const double m_dCxP;
    const double m_dCyP;
    const double m_dCxNormalized;
    const double m_dCyNormalized;
    const double m_dFocalLengthMeters;
    const Eigen::Vector4d m_vecDistortionCoefficients;
    const Eigen::Matrix3d m_matRectification;
    const Eigen::Vector2d m_vecPrincipalPoint;
    const Eigen::Vector3d m_vecPrincipalPointNormalized;

    //ds misc
    const int32_t m_iWidthPixel;
    const int32_t m_iHeightPixel;
    const double m_dWidthPixels;
    const double m_dHeightPixels;
    const float m_fWidthPixels;
    const float m_fHeightPixels;
    const std::pair< double, double > m_prRangeWidthNormalized;
    const std::pair< double, double > m_prRangeHeightNormalized;
    const cv::Rect m_cFieldOfView;

//ds access
public:

    const Eigen::Vector3d getNormalHomogenized( const Eigen::Vector2d& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint(0)-m_dCxP )/m_dFxP, ( p_vecPoint(1)-m_dCyP )/m_dFyP, 1.0 );
    }
    const Eigen::Vector3d getNormalHomogenized( const cv::KeyPoint& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.pt.x-m_dCxP )/m_dFxP, ( p_vecPoint.pt.y-m_dCyP )/m_dFyP, 1.0 );
    }
    const Eigen::Vector3d getNormalHomogenized( const cv::Point2d& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCxP )/m_dFxP, ( p_vecPoint.y-m_dCyP )/m_dFyP, 1.0 );
    }
    const Eigen::Vector3d getNormalHomogenized( const cv::Point2f& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCxP )/m_dFxP, ( p_vecPoint.y-m_dCyP )/m_dFyP, 1.0 );
    }
    const Eigen::Vector3d getNormalHomogenized( const cv::Point2i& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCxP )/m_dFxP, ( p_vecPoint.y-m_dCyP )/m_dFyP, 1.0 );
    }
    const Eigen::Vector2d getNormalized( const cv::Point2i& p_vecPoint ) const
    {
        return Eigen::Vector2d( ( p_vecPoint.x-m_dCxP )/m_dFxP, ( p_vecPoint.y-m_dCyP )/m_dFyP );
    }
    const Eigen::Vector2d getNormalized( const cv::Point2f& p_vecPoint ) const
    {
        return Eigen::Vector2d( ( p_vecPoint.x-m_dCxP )/m_dFxP, ( p_vecPoint.y-m_dCyP )/m_dFyP );
    }
    const Eigen::Vector2d getNormalized( const cv::Point2d& p_vecPoint ) const
    {
        return Eigen::Vector2d( ( p_vecPoint.x-m_dCxP )/m_dFxP, ( p_vecPoint.y-m_dCyP )/m_dFyP );
    }
    const double getNormalizedX( const double& p_dX ) const
    {
        return ( p_dX-m_dCxP )/m_dFxP;
    }
    const double getNormalizedY( const double& p_dY ) const
    {
        return ( p_dY-m_dCyP )/m_dFyP;
    }
    const cv::Point2d getDenormalized( const Eigen::Vector2d& p_vecPoint ) const
    {
        return cv::Point2d( p_vecPoint(0)*m_dFxP+m_dCxP, p_vecPoint(1)*m_dFyP+m_dCyP );
    }
    const double getDenormalizedX( const double& p_dX ) const
    {
        return p_dX*m_dFxP+m_dCxP;
    }
    const double getDenormalizedY( const double& p_dY ) const
    {
        return p_dY*m_dFyP+m_dCyP;
    }
    const int32_t getU( const double& p_dX ) const
    {
        return p_dX*m_dFxP+m_dCxP;
    }
    const int32_t getV( const double& p_dY ) const
    {
        return p_dY*m_dFyP+m_dCyP;
    }
    const CPoint2DHomogenized getHomogeneousProjection( const CPoint3DHomogenized& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*p_vecPoint );

        //ds return homogeneous
        return vecProjectionInhomogeneous/vecProjectionInhomogeneous(2);
    }
    const CPoint2DHomogenized getHomogeneousProjection( const CPoint3DCAMERA& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPoint(0), p_vecPoint(1), p_vecPoint(2), 1.0 ) );

        //ds return homogeneous
        return vecProjectionInhomogeneous/vecProjectionInhomogeneous(2);
    }
    const cv::Point2d getProjection( const CPoint3DCAMERA& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPoint(0), p_vecPoint(1), p_vecPoint(2), 1.0 ) );

        //ds return uv point
        return cv::Point2d( vecProjectionInhomogeneous(0)/vecProjectionInhomogeneous(2), vecProjectionInhomogeneous(1)/vecProjectionInhomogeneous(2) );
    }
    const cv::Point2f getProjectionRounded( const CPoint3DCAMERA& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPoint(0), p_vecPoint(1), p_vecPoint(2), 1.0 ) );

        //ds return rounded uv point
        return cv::Point2f( std::round( static_cast< float >( vecProjectionInhomogeneous(0)/vecProjectionInhomogeneous(2) ) ),
                            std::round( static_cast< float >( vecProjectionInhomogeneous(1)/vecProjectionInhomogeneous(2) ) ) );
    }
    const cv::Point2f getUV( const CPoint3DCAMERA& p_vecPointXYZ ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPointXYZ.x( ), p_vecPointXYZ.y( ), p_vecPointXYZ.z( ), 1.0 ) );

        //ds return uv point
        return cv::Point2d( vecProjectionInhomogeneous(0)/vecProjectionInhomogeneous(2), vecProjectionInhomogeneous(1)/vecProjectionInhomogeneous(2) );
    }

    const double getPrincipalWeightU( const cv::Point2d& p_ptUV ) const
    {
        return std::sqrt( std::fabs( p_ptUV.x-m_dCxP ) )/10.0;
    }
    const double getPrincipalWeightV( const cv::Point2d& p_ptUV ) const
    {
        return std::sqrt( std::fabs( p_ptUV.y-m_dCyP ) )/10.0;
    }

//ds helpers
private:

    virtual void _logConfiguration( const double& p_dImprecision ) const
    {
        //ds log complete configuration
        CLogger::openBox( );
        std::cout << "Configuration camera: " << m_strCameraLabel << "\n\n"
                  << "Configuration imprecision: " << p_dImprecision << "\n"
                  << "FxP: " << m_dFxP << "\n"
                  << "FyP: " << m_dFyP << "\n"
                  << "CxP: " << m_dCxP << "\n"
                  << "CyP: " << m_dCyP << "\n"
                  << "\nIntrinsic matrix (K):\n\n" << m_matIntrinsic << "\n\n"
                  << "Distortion coefficients (D): " << m_vecDistortionCoefficients.transpose( ) << "\n"
                  << "\nRectification matrix (R):\n\n" << m_matRectification << "\n\n"
                  << "\nProjection matrix (P):\n\n" << m_matProjection << "\n\n"
                  << "Principal point: " << m_vecPrincipalPoint.transpose( ) << "\n"
                  << "Resolution (w x h): " << m_uWidthPixel << " x " << m_uHeightPixel << "\n"
                  << "Normalized x range: [" << m_prRangeWidthNormalized.first << ", " << m_prRangeWidthNormalized.second << "]\n"
                  << "Normalized y range: [" << m_prRangeHeightNormalized.first << ", " << m_prRangeHeightNormalized.second << "]" << std::endl;
        CLogger::closeBox( );
    }

};


#endif //CPINHOLECAMERA_H
