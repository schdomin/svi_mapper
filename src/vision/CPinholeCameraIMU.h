#ifndef CPINHOLECAMERAIMU_H
#define CPINHOLECAMERAIMU_H

#include <iostream>

#include "types/Types.h"
#include "CPinholeCamera.h"
#include "utility/CLogger.h"



class CPinholeCameraIMU: public CPinholeCamera
{

public:

    //ds construct from parameter vector
    CPinholeCameraIMU( const std::string& p_strLabel,
                    const uint32_t& p_uWidthPixels,
                    const uint32_t& p_uHeightPixels,
                    const MatrixProjection& p_matProjection,
                    const Eigen::Matrix3d& p_matIntrinsic,
                    const double& p_dFocalLengthMeters,
                    const Eigen::Vector4d& p_vecDistortionCoefficients,
                    const Eigen::Matrix3d& p_matRectification,
                    const Eigen::Quaterniond& p_vecRotationToIMUInitial,
                    const Eigen::Vector3d& p_vecTranslationToIMUInitial,
                    const Eigen::Matrix3d& p_matRotationCorrectionCAMERAtoIMU ): CPinholeCamera( p_strLabel,
                                                                                p_uWidthPixels,
                                                                                p_uHeightPixels,
                                                                                p_matProjection,
                                                                                p_matIntrinsic,
                                                                                p_dFocalLengthMeters,
                                                                                p_vecDistortionCoefficients,
                                                                                p_matRectification ),
                                                                                m_vecRotationToIMUInitial( p_vecRotationToIMUInitial ),
                                                                                m_matRotationToIMUInitial( m_vecRotationToIMUInitial.toRotationMatrix( ) ),
                                                                                m_vecTranslationToIMUInitial( p_vecTranslationToIMUInitial ),
                                                                                m_matRotationCorrectionCAMERAtoIMU( p_matRotationCorrectionCAMERAtoIMU ),
                                                                                m_matRotationCAMERAtoIMU( m_matRotationCorrectionCAMERAtoIMU*m_matRotationToIMUInitial ),
                                                                                m_matRotationIMUtoCAMERA( m_matRotationCAMERAtoIMU.inverse( ) ),
                                                                                m_matTransformationCAMERAtoIMU( Eigen::Matrix4d::Identity( ) ),
                                                                                m_matTransformationIMUtoCAMERA( Eigen::Matrix4d::Identity( ) )
    {
        //ds compute the rotated transformation
        m_matTransformationCAMERAtoIMU.linear( )      = m_matRotationCAMERAtoIMU;
        m_matTransformationCAMERAtoIMU.translation( ) = m_vecTranslationToIMUInitial;

        //ds precompute inverse
        m_matTransformationIMUtoCAMERA = m_matTransformationCAMERAtoIMU.inverse( );

        //ds buffer rotation matrix for check
        const Eigen::Matrix3d matRotationCheck( m_matTransformationIMUtoCAMERA.linear( ) );

        //ds coefficients difference
        double dDifferenceCoefficients = 0;

        //ds check configuration
        for( uint8_t u = 0; u < 3; ++u )
        {
            for( uint8_t v = 0; v < 3; ++v )
            {
                dDifferenceCoefficients += std::fabs( static_cast< double >( m_matRotationIMUtoCAMERA(u,v)-matRotationCheck(u,v) ) );
            }
        }
        assert( dDifferenceCoefficients < 1e-5 );

        //ds log complete configuration
        //_logConfiguration( dDifferenceCoefficients );
    }

    //ds no manual dynamic allocation
    ~CPinholeCameraIMU( ){ }

public:

    //ds extrinsics
    const Eigen::Quaterniond m_vecRotationToIMUInitial;
    const Eigen::Matrix3d m_matRotationToIMUInitial;
    const Eigen::Vector3d m_vecTranslationToIMUInitial;
    const Eigen::Matrix3d m_matRotationCorrectionCAMERAtoIMU;
    const Eigen::Matrix3d m_matRotationCAMERAtoIMU;
    const Eigen::Matrix3d m_matRotationIMUtoCAMERA;
    Eigen::Isometry3d m_matTransformationCAMERAtoIMU;
    Eigen::Isometry3d m_matTransformationIMUtoCAMERA;

//ds helpers
private:

    void _logConfiguration( const double& p_dImprecision ) const
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
                  << "Initial quaternion x y z w (CAMERA to IMU): " << m_vecRotationToIMUInitial.x( ) << " " << m_vecRotationToIMUInitial.y( ) << " " << m_vecRotationToIMUInitial.z( ) << " " << m_vecRotationToIMUInitial.w( ) << "\n"
                  << "Initial translation (CAMERA to IMU): " << m_vecTranslationToIMUInitial.transpose( ) << "\n"
                  << "\nRotation matrix correction (CAMERA to IMU):\n" << m_matRotationCorrectionCAMERAtoIMU << "\n"
                  << "\nTransformation matrix (CAMERA to IMU):\n\n" << m_matTransformationCAMERAtoIMU.matrix( ) << "\n\n"
                  << "\nTransformation matrix (IMU to CAMERA):\n\n" << m_matTransformationIMUtoCAMERA.matrix( ) << "\n\n"
                  << "Resolution (w x h): " << m_uWidthPixel << " x " << m_uHeightPixel << "\n"
                  << "Normalized x range: [" << m_prRangeWidthNormalized.first << ", " << m_prRangeWidthNormalized.second << "]\n"
                  << "Normalized y range: [" << m_prRangeHeightNormalized.first << ", " << m_prRangeHeightNormalized.second << "]" << std::endl;
        CLogger::closeBox( );
    }

};


#endif //CPINHOLECAMERAIMU_H
