#include <iostream>
#include <string>
#include <math.h>
#include <opencv/cv.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

//ds custom
#include "utility/CParameterBase.h"
#include "vision/CMiniVisionToolbox.h"



int32_t main( int32_t argc, char **argv )
{
    //ds configuration
    std::string strInfileMessageDump        = "";
    std::string strConfigurationCameraLEFT  = "../hardware_parameters/vi_sensor_camera_left.txt";
    std::string strConfigurationCameraRIGHT = "../hardware_parameters/vi_sensor_camera_right.txt";

    try
    {
        //ds load camera parameters
        CParameterBase::loadCameraLEFT( strConfigurationCameraLEFT );
        std::printf( "(main) successfully imported camera LEFT\n" );
        CParameterBase::loadCameraRIGHT( strConfigurationCameraRIGHT );
        std::printf( "(main) successfully imported camera RIGHT\n" );
    }
    catch( const CExceptionParameter& p_cException )
    {
        std::printf( "(main) unable to import camera parameters - CExceptionParameter: '%s'\n", p_cException.what( ) );
        std::fflush( stdout );
        return 1;
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "(main) unable to import camera parameters - std::invalid_argument: '%s'\n", p_cException.what( ) );
        std::fflush( stdout );
        return 1;
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "(main) unable to import camera parameters - std::out_of_range: '%s'\n", p_cException.what( ) );
        std::fflush( stdout );
        return 1;
    }

    //ds projection matrices parameters
    const double dFL  = CParameterBase::pCameraLEFT->m_matProjection(0,0);
    const double dPuL = CParameterBase::pCameraLEFT->m_matProjection(0,2);
    const double dDuL = CParameterBase::pCameraLEFT->m_matProjection(0,3);
    const double dPvL = CParameterBase::pCameraLEFT->m_matProjection(1,2);
    const double dDvL = CParameterBase::pCameraLEFT->m_matProjection(1,3);
    const double dFR  = CParameterBase::pCameraRIGHT->m_matProjection(0,0);
    const double dPuR = CParameterBase::pCameraRIGHT->m_matProjection(0,2);
    const double dDuR = CParameterBase::pCameraRIGHT->m_matProjection(0,3);
    const double dPvR = CParameterBase::pCameraRIGHT->m_matProjection(1,2);
    const double dDvR = CParameterBase::pCameraRIGHT->m_matProjection(1,3);

    std::printf( "\ndFL: %f\n", dFL );
    std::printf( "dPuL: %f\n", dPuL );
    std::printf( "dDuL: %f\n", dDuL );
    std::printf( "dPvL: %f\n", dPvL );
    std::printf( "dDvL: %f\n", dDvL );

    std::printf( "\ndFR: %f\n", dFR );
    std::printf( "dPuR: %f\n", dPuR );
    std::printf( "dDuR: %f\n", dDuR );
    std::printf( "dPvR: %f\n", dPvR );
    std::printf( "dDvR: %f\n", dDvR );


    //ds enforce epipolar situation
    assert( 1.0 == dFL/dFR );
    assert( 0.0 == dDvL );
    assert( 0.0 == dDvR );

    //ds sampling range 0 to
    const uint8_t uSampleLimit = 75;

    //ds coordinates
    const cv::Point2f ptUVLEFT( 400, 200 );
    const cv::Point2f ptUVRIGHTMinimum( 0, ptUVLEFT.y );

    //ds compute minimum depth
    const CPoint3DCAMERA vecPointXYZMinimum( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( ptUVLEFT,
                                                                                                         ptUVRIGHTMinimum,
                                                                                                         CParameterBase::pCameraLEFT->m_matProjection,
                                                                                                         CParameterBase::pCameraRIGHT->m_matProjection ) );

    //ds compute exponent for minimal depth
    const int8_t iExponentShift = std::ceil( log( vecPointXYZMinimum.z( ) )/log( 1.1 ) );
    std::printf( "\nminimum depth: %f -> exponent: %i\n\n", vecPointXYZMinimum.z( ), iExponentShift );

    //ds previous point u (for break condition)
    int32_t iUPrevious = -1;

    //ds sample depth
    for( uint8_t u = 0; u < uSampleLimit; ++u )
    {
        //ds sample 3D point in camera frame
        const double dZ = pow( 1.1, u+iExponentShift );

        //ds compute projection point directly
        const cv::Point2i ptUVRIGHT( ( ptUVLEFT.x-dPuL-dDuL/dZ ) + dPuR + dDuR/dZ,
                                     ( ptUVLEFT.y-dPvL-dDvL/dZ ) + dPvR + dDvR/dZ );
        assert( 0 <= ptUVRIGHT.x );
        assert( 0 <= ptUVRIGHT.y );

        //ds check if precision converged
        if( iUPrevious == ptUVRIGHT.x )
        {
            //ds keep going
        }
        else
        {
            std::printf( "[%02u] depth: %6.2f - UV: [%3i][%3i]\n", u, dZ, ptUVRIGHT.x, ptUVRIGHT.y );
            iUPrevious = ptUVRIGHT.x;
        }
    }


    return 0;
}
