#ifndef CSTEREOCAMERA_H
#define CSTEREOCAMERA_H

#include "CPinholeCamera.h"
#include "utility/CWrapperOpenCV.h"



class CStereoCamera
{

public:

    CStereoCamera( const std::shared_ptr< CPinholeCamera > p_pCameraLEFT,
                   const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT,
                   const Eigen::Vector3d& p_vecTranslationToRIGHT ): m_pCameraLEFT( p_pCameraLEFT ),
                                                                     m_pCameraRIGHT( p_pCameraRIGHT ),
                                                                   m_uPixelWidth( p_pCameraLEFT->m_uWidthPixel ),
                                                                   m_uPixelHeight( p_pCameraLEFT->m_uHeightPixel ),
                                                                   m_fWidthPixels( m_uPixelWidth ),
                                                                   m_fHeightPixels( m_uPixelHeight ),
                                                                   m_cVisibleRange( 28, 28, m_uPixelWidth-56, m_uPixelHeight-56 )
    {
        //ds adjust transformation
        m_matTransformLEFTtoRIGHT = Eigen::Matrix4d::Identity( );
        m_matTransformLEFTtoRIGHT.translation( ) = p_vecTranslationToRIGHT;

        m_dBaselineMeters = m_matTransformLEFTtoRIGHT.translation( ).norm( );

        CLogger::openBox( );
        std::printf( "<CStereoCamera>(CStereoCamera) manually set transformation LEFT to RIGHT: \n\n" );
        std::cout << m_matTransformLEFTtoRIGHT.matrix( ) << "\n" << std::endl;
        std::printf( "<CStereoCamera>(CStereoCamera) new baseline: %f\n", m_dBaselineMeters );
        CLogger::closeBox( );
    }

    //ds wrapping constructors
    CStereoCamera( const CPinholeCamera& p_cCameraLEFT,
                   const CPinholeCamera& p_cCameraRIGHT,
                   const Eigen::Vector3d& p_vecTranslationToRIGHT ): CStereoCamera( std::make_shared< CPinholeCamera >( p_cCameraLEFT ),
                                                                     std::make_shared< CPinholeCamera >( p_cCameraRIGHT ),
                                                                     p_vecTranslationToRIGHT )
    {
        //ds nothing to do
    }

    CStereoCamera( const std::shared_ptr< CPinholeCamera > p_pCameraLEFT,
                   const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT,
                   const Eigen::Isometry3d& p_matTranslationToRIGHT ): m_pCameraLEFT( p_pCameraLEFT ),
                                                                       m_pCameraRIGHT( p_pCameraRIGHT ),
                                                                       m_uPixelWidth( p_pCameraLEFT->m_uWidthPixel ),
                                                                       m_uPixelHeight( p_pCameraLEFT->m_uHeightPixel ),
                                                                       m_fWidthPixels( m_uPixelWidth ),
                                                                       m_fHeightPixels( m_uPixelHeight ),
                                                                       m_cVisibleRange( 28, 28, m_uPixelWidth-56, m_uPixelHeight-56 )
    {
        //ds adjust transformation
        m_matTransformLEFTtoRIGHT = p_matTranslationToRIGHT;

        m_dBaselineMeters = m_matTransformLEFTtoRIGHT.translation( ).norm( );

        CLogger::openBox( );
        std::printf( "<CStereoCamera>(CStereoCamera) manually set transformation LEFT to RIGHT: \n\n" );
        std::cout << m_matTransformLEFTtoRIGHT.matrix( ) << "\n" << std::endl;
        std::printf( "<CStereoCamera>(CStereoCamera) new baseline: %f\n", m_dBaselineMeters );
        CLogger::closeBox( );
    }

    //ds no manual dynamic allocation
    virtual ~CStereoCamera( ){ }

//ds fields
public:

    //ds stereo cameras
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;

    //ds intrinsics
    double m_dBaselineMeters;

    //ds common dimensions
    const uint32_t m_uPixelWidth;
    const uint32_t m_uPixelHeight;
    const float m_fWidthPixels;
    const float m_fHeightPixels;

    //ds extrinsics
    Eigen::Isometry3d m_matTransformLEFTtoRIGHT;

    //ds undistortion/rectification
    cv::Mat m_arrUndistortRectifyMapsLEFT[2];
    cv::Mat m_arrUndistortRectifyMapsRIGHT[2];

    //ds visible range
    const cv::Rect m_cVisibleRange;

//ds accessors
public:

    //ds undistortion/rectification (TODO remove UGLY in/out)
    void undistortAndrectify( cv::Mat& p_matImageLEFT, cv::Mat& p_matImageRIGHT ) const
    {
        //ds remap images
        cv::remap( p_matImageLEFT, p_matImageLEFT, m_arrUndistortRectifyMapsLEFT[0], m_arrUndistortRectifyMapsLEFT[1], cv::INTER_LINEAR );
        cv::remap( p_matImageRIGHT, p_matImageRIGHT, m_arrUndistortRectifyMapsRIGHT[0], m_arrUndistortRectifyMapsRIGHT[1], cv::INTER_LINEAR );
    }

};

#endif //CSTEREOCAMERA_H
