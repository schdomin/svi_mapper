#ifndef CSTEREOCAMERAIMU_H
#define CSTEREOCAMERAIMU_H

#include "CStereoCamera.h"
#include "CPinholeCameraIMU.h"
#include "utility/CWrapperOpenCV.h"



class CStereoCameraIMU: public CStereoCamera
{

public:

    CStereoCameraIMU( const std::shared_ptr< CPinholeCameraIMU > p_pCameraLEFT,
                      const std::shared_ptr< CPinholeCameraIMU > p_pCameraRIGHT ): CStereoCamera( p_pCameraLEFT, p_pCameraRIGHT, Eigen::Vector3d::Zero( ) ),
                                                                                   m_pCameraLEFT( p_pCameraLEFT ),
                                                                                   m_pCameraRIGHT( p_pCameraRIGHT )
    {
        //ds setup extrinsic transformations
        m_matTransformLEFTtoIMU   = p_pCameraLEFT->m_matTransformationCAMERAtoIMU;
        m_matTransformRIGHTtoIMU  = p_pCameraRIGHT->m_matTransformationCAMERAtoIMU;
        m_matTransformLEFTtoRIGHT = m_matTransformRIGHTtoIMU.inverse( )*m_matTransformLEFTtoIMU;

        m_dBaselineMeters = m_matTransformLEFTtoRIGHT.translation( ).norm( );

        //ds log complete configuration
        CLogger::openBox( );
        std::printf( "<CStereoCameraIMU>(CStereoCameraIMU) configuration stereo camera IMU: '%s'-IMU-'%s' \n\n", m_pCameraLEFT->m_strCameraLabel.c_str( ), m_pCameraRIGHT->m_strCameraLabel.c_str( ) );
        std::cout << m_matTransformLEFTtoRIGHT.matrix( ) << "\n" << std::endl;
        std::printf( "<CStereoCameraIMU>(CStereoCameraIMU) new baseline: %f\n", m_dBaselineMeters );
        CLogger::closeBox( );

        //ds compute undistorted and rectified mappings
        cv::initUndistortRectifyMap( CWrapperOpenCV::toCVMatrix( p_pCameraLEFT->m_matIntrinsic ),
                                     CWrapperOpenCV::toCVVector( p_pCameraLEFT->m_vecDistortionCoefficients ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraLEFT->m_matRectification ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraLEFT->m_matProjection ),
                                     cv::Size( m_pCameraLEFT->m_uWidthPixel, m_pCameraLEFT->m_uHeightPixel ),
                                     CV_16SC2,
                                     m_arrUndistortRectifyMapsLEFT[0],
                                     m_arrUndistortRectifyMapsLEFT[1] );
        cv::initUndistortRectifyMap( CWrapperOpenCV::toCVMatrix( p_pCameraRIGHT->m_matIntrinsic ),
                                     CWrapperOpenCV::toCVVector( p_pCameraRIGHT->m_vecDistortionCoefficients ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraRIGHT->m_matRectification ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraRIGHT->m_matProjection ),
                                     cv::Size( m_pCameraRIGHT->m_uWidthPixel, m_pCameraRIGHT->m_uHeightPixel ),
                                     CV_16SC2,
                                     m_arrUndistortRectifyMapsRIGHT[0],
                                     m_arrUndistortRectifyMapsRIGHT[1] );
    }

    //ds wrapping constructors
    CStereoCameraIMU( const CPinholeCameraIMU& p_cCameraLEFT, const CPinholeCameraIMU& p_cCameraRIGHT ): CStereoCameraIMU( std::make_shared< CPinholeCameraIMU >( p_cCameraLEFT ), std::make_shared< CPinholeCameraIMU >( p_cCameraRIGHT ) )
    {
        //ds nothing to do
    }

    //ds no manual dynamic allocation
    ~CStereoCameraIMU( ){ }

//ds fields
public:

    //ds stereo cameras
    const std::shared_ptr< CPinholeCameraIMU > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCameraIMU > m_pCameraRIGHT;

    //ds extrinsics
    Eigen::Isometry3d m_matTransformLEFTtoIMU;
    Eigen::Isometry3d m_matTransformRIGHTtoIMU;
    Eigen::Isometry3d m_matTransformLEFTtoRIGHT;

};

#endif //CSTEREOCAMERAIMU_H
