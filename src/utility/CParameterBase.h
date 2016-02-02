#ifndef CPARAMETERBASE_H
#define CPARAMETERBASE_H

#include <fstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

//ds custom
#include "../vision/CStereoCameraIMU.h"
#include "exceptions/CExceptionParameter.h"



class CParameterBase
{

public:

    //ds vectorize camera configuration file parameters
    static const std::vector< std::string > getParametersFromFile( const std::string& p_strCameraConfigurationFile )
    {
        std::vector< std::string > vecParameters;

        //ds open the file for reading
        std::ifstream ifConfiguration( p_strCameraConfigurationFile, std::ifstream::in );

        //ds check failure
        if( !ifConfiguration.is_open( ) || ifConfiguration.bad( ) )
        {
            throw CExceptionParameter( "unable to open file: '" + p_strCameraConfigurationFile + "'" );
        }

        //ds start parsing
        std::string strLineBuffer;
        while( std::getline( ifConfiguration, strLineBuffer ) )
        {
            //ds process string if not empty
            if( !strLineBuffer.empty( ) )
            {
                std::string::size_type uLastStart( 0 );
                std::string::size_type uLastSeparator( 0 );
                uLastSeparator = strLineBuffer.find( ' ', uLastStart );

                //ds look for spaces
                while( std::string::npos != uLastSeparator )
                {
                    //ds store previous element
                    vecParameters.push_back( strLineBuffer.substr( uLastStart, uLastSeparator-uLastStart ) );
                    uLastStart = uLastSeparator+1;
                    uLastSeparator = strLineBuffer.find( ' ', uLastStart );
                }

                //ds store last element
                vecParameters.push_back( strLineBuffer.substr( uLastStart ) );
            }
        }

        //for( const std::string& strParameter: vecParameters )
        //{
        //    std::printf( "(getParametersFromFile) value: '%s'\n", strParameter.c_str( ) );
        //}
        //std::printf( "(getParametersFromFile) total values: %lu\n", vecParameters.size( ) );

        return vecParameters;
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static const double getDoubleFromFile( const std::vector< std::string >& p_vecParameters, const std::string& p_strParameterName )
    {
        //ds look for the parameter
        const std::vector< std::string >::const_iterator itParameter( std::find( p_vecParameters.begin( ), p_vecParameters.end( ), p_strParameterName.c_str( ) ) );

        //ds escape if not found
        if( p_vecParameters.end( ) == itParameter )
        {
            throw CExceptionParameter( "cannot find parameter: " + p_strParameterName );
        }

        //ds parse the next value (non-catch intended)
        const double dValue = std::stod( *( itParameter+1 ) );

        //ds return if still here
        return dValue;
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static const uint32_t getIntegerFromFile( const std::vector< std::string >& p_vecParameters, const std::string& p_strParameterName )
    {
        //ds look for the parameter
        const std::vector< std::string >::const_iterator itParameter( std::find( p_vecParameters.begin( ), p_vecParameters.end( ), p_strParameterName.c_str( ) ) );

        //ds escape if not found
        if( p_vecParameters.end( ) == itParameter )
        {
            throw CExceptionParameter( "cannot find parameter: " + p_strParameterName );
        }

        //ds parse the next value (non-catch intended)
        const uint32_t uValue = std::stoul( *( itParameter+1 ) );

        //ds return if still here
        return uValue;
    }

    template < uint32_t uRows, uint32_t uCols >
    static const Eigen::Matrix< double, uRows, uCols > getMatrixFromFile( const std::vector< std::string >& p_vecParameters, const std::string& p_strParameterName )
    {
        assert( 0 < uRows );
        assert( 0 < uCols );

        //ds look for the parameter
        const std::vector< std::string >::const_iterator itParameter( std::find( p_vecParameters.begin( ), p_vecParameters.end( ), p_strParameterName.c_str( ) ) );

        //ds escape if not found
        if( p_vecParameters.end( ) == itParameter )
        {
            throw CExceptionParameter( "cannot find parameter: " + p_strParameterName );
        }

        //ds allocate an empty matrix
        Eigen::Matrix< double, uRows, uCols > matMatrix( Eigen::Matrix< double, uRows, uCols >::Zero( ) );

        //ds position
        uint32_t uElementCount = 0;

        //ds for each row
        for( uint32_t u = 0; u < uRows; ++u )
        {
            //ds loop over columns
            for( uint32_t v = 0; v < uCols; ++v )
            {
                //ds set the element
                matMatrix(u,v) = std::stod( *( itParameter+u*uCols+v+1 ) );
                ++uElementCount;
            }
        }

        //ds return filled matrix
        return matMatrix;
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static const Eigen::Quaterniond getQuanternionFromFile( const std::vector< std::string >& p_vecParameters, const std::string& p_strParameterName )
    {
        //ds look for the parameter
        const std::vector< std::string >::const_iterator itParameter( std::find( p_vecParameters.begin( ), p_vecParameters.end( ), p_strParameterName.c_str( ) ) );

        //ds escape if not found
        if( p_vecParameters.end( ) == itParameter )
        {
            throw CExceptionParameter( "cannot find parameter: " + p_strParameterName );
        }

        //ds empty quaternion
        Eigen::Quaterniond vecQuaternion( 1.0, 0.0, 0.0, 0.0 );

        //ds fill in the values: x,y,z,w - exception flythru intended
        vecQuaternion.x( ) = std::stod( *( itParameter+1 ) );
        vecQuaternion.y( ) = std::stod( *( itParameter+2 ) );
        vecQuaternion.z( ) = std::stod( *( itParameter+3 ) );
        vecQuaternion.w( ) = std::stod( *( itParameter+4 ) );

        //ds return result
        return vecQuaternion;
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static void loadCameraLEFT( const std::string& p_strCameraConfigurationFile )
    {
        //ds params
        const std::vector< std::string > vecParameters( getParametersFromFile( p_strCameraConfigurationFile ) );

        //ds parse core parameters
        const std::string strCameraLabel( vecParameters.front( ) );
        const uint32_t m_uWidthPixel  = CParameterBase::getIntegerFromFile( vecParameters, "uWidthPixels" );
        const uint32_t m_uHeightPixel = CParameterBase::getIntegerFromFile( vecParameters, "uHeightPixels" );

        const MatrixProjection matProjection( CParameterBase::getMatrixFromFile< 3, 4 >( vecParameters, "matProjection" ) );
        const Eigen::Matrix3d matIntrinsic( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matIntrinsic" ) );

        const double dFocalLengthMeters = CParameterBase::getDoubleFromFile( vecParameters, "dFocalLengthMeters" );
        const Eigen::Vector4d vecDistortionCoefficients( CParameterBase::getMatrixFromFile< 4, 1 >( vecParameters, "vecDistortionCoefficients" ) );
        const Eigen::Matrix3d matRectification( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matRectification" ) );

        //ds create camera instance
        pCameraLEFT = std::make_shared< CPinholeCamera >( strCameraLabel,
                                                          m_uWidthPixel,
                                                          m_uHeightPixel,
                                                          matProjection,
                                                          matIntrinsic,
                                                          dFocalLengthMeters,
                                                          vecDistortionCoefficients,
                                                          matRectification );
        assert( 0 != pCameraLEFT );
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static void loadCameraRIGHT( const std::string& p_strCameraConfigurationFile )
    {
        //ds params
        const std::vector< std::string > vecParameters( getParametersFromFile( p_strCameraConfigurationFile ) );

        //ds parse core parameters
        const std::string strCameraLabel( vecParameters.front( ) );
        const uint32_t m_uWidthPixel  = CParameterBase::getIntegerFromFile( vecParameters, "uWidthPixels" );
        const uint32_t m_uHeightPixel = CParameterBase::getIntegerFromFile( vecParameters, "uHeightPixels" );

        const MatrixProjection matProjection( CParameterBase::getMatrixFromFile< 3, 4 >( vecParameters, "matProjection" ) );
        const Eigen::Matrix3d matIntrinsic( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matIntrinsic" ) );

        const double dFocalLengthMeters = CParameterBase::getDoubleFromFile( vecParameters, "dFocalLengthMeters" );
        const Eigen::Vector4d vecDistortionCoefficients( CParameterBase::getMatrixFromFile< 4, 1 >( vecParameters, "vecDistortionCoefficients" ) );
        const Eigen::Matrix3d matRectification( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matRectification" ) );

        //ds create camera instance
        pCameraRIGHT = std::make_shared< CPinholeCamera >( strCameraLabel,
                                                          m_uWidthPixel,
                                                          m_uHeightPixel,
                                                          matProjection,
                                                          matIntrinsic,
                                                          dFocalLengthMeters,
                                                          vecDistortionCoefficients,
                                                          matRectification );
        assert( 0 != pCameraRIGHT );
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static void loadCameraLEFTwithIMU( const std::string& p_strCameraConfigurationFile )
    {
        //ds params
        const std::vector< std::string > vecParameters( getParametersFromFile( p_strCameraConfigurationFile ) );

        //ds parse core parameters
        const std::string strCameraLabel( vecParameters.front( ) );
        const uint32_t m_uWidthPixel  = CParameterBase::getIntegerFromFile( vecParameters, "uWidthPixels" );
        const uint32_t m_uHeightPixel = CParameterBase::getIntegerFromFile( vecParameters, "uHeightPixels" );

        const MatrixProjection matProjection( CParameterBase::getMatrixFromFile< 3, 4 >( vecParameters, "matProjection" ) );
        const Eigen::Matrix3d matIntrinsic( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matIntrinsic" ) );

        const double dFocalLengthMeters = CParameterBase::getDoubleFromFile( vecParameters, "dFocalLengthMeters" );
        const Eigen::Vector4d vecDistortionCoefficients( CParameterBase::getMatrixFromFile< 4, 1 >( vecParameters, "vecDistortionCoefficients" ) );
        const Eigen::Matrix3d matRectification( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matRectification" ) );

        const Eigen::Quaterniond vecRotationToIMUInitial( CParameterBase::getQuanternionFromFile( vecParameters, "vecQuaternionToIMU" ) );
        const Eigen::Vector3d vecTranslationToIMUInitial( CParameterBase::getMatrixFromFile< 3, 1 >( vecParameters, "vecTranslationToIMU" ) );
        const Eigen::Matrix3d matRotationCorrectionCAMERAtoIMU( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matRotationIntrinsicCAMERAtoIMU" ) );

        //ds create camera instance
        pCameraLEFTwithIMU = std::make_shared< CPinholeCameraIMU >( strCameraLabel,
                                                          m_uWidthPixel,
                                                          m_uHeightPixel,
                                                          matProjection,
                                                          matIntrinsic,
                                                          dFocalLengthMeters,
                                                          vecDistortionCoefficients,
                                                          matRectification,
                                                          vecRotationToIMUInitial,
                                                          vecTranslationToIMUInitial,
                                                          matRotationCorrectionCAMERAtoIMU );
        assert( 0 != pCameraLEFTwithIMU );
    }

    //ds THROWS: CExceptionParameter, std::invalid_argument, std::out_of_range
    static void loadCameraRIGHTwithIMU( const std::string& p_strCameraConfigurationFile )
    {
        //ds params
        const std::vector< std::string > vecParameters( getParametersFromFile( p_strCameraConfigurationFile ) );

        //ds parse core parameters
        const std::string strCameraLabel( vecParameters.front( ) );
        const uint32_t m_uWidthPixel  = CParameterBase::getIntegerFromFile( vecParameters, "uWidthPixels" );
        const uint32_t m_uHeightPixel = CParameterBase::getIntegerFromFile( vecParameters, "uHeightPixels" );

        const MatrixProjection matProjection( CParameterBase::getMatrixFromFile< 3, 4 >( vecParameters, "matProjection" ) );
        const Eigen::Matrix3d matIntrinsic( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matIntrinsic" ) );

        const double dFocalLengthMeters = CParameterBase::getDoubleFromFile( vecParameters, "dFocalLengthMeters" );
        const Eigen::Vector4d vecDistortionCoefficients( CParameterBase::getMatrixFromFile< 4, 1 >( vecParameters, "vecDistortionCoefficients" ) );
        const Eigen::Matrix3d matRectification( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matRectification" ) );

        const Eigen::Quaterniond vecRotationToIMUInitial( CParameterBase::getQuanternionFromFile( vecParameters, "vecQuaternionToIMU" ) );
        const Eigen::Vector3d vecTranslationToIMUInitial( CParameterBase::getMatrixFromFile< 3, 1 >( vecParameters, "vecTranslationToIMU" ) );
        const Eigen::Matrix3d matRotationCorrectionCAMERAtoIMU( CParameterBase::getMatrixFromFile< 3, 3 >( vecParameters, "matRotationIntrinsicCAMERAtoIMU" ) );

        //ds create camera instance
        pCameraRIGHTwithIMU = std::make_shared< CPinholeCameraIMU >( strCameraLabel,
                                                           m_uWidthPixel,
                                                           m_uHeightPixel,
                                                           matProjection,
                                                           matIntrinsic,
                                                           dFocalLengthMeters,
                                                           vecDistortionCoefficients,
                                                           matRectification,
                                                           vecRotationToIMUInitial,
                                                           vecTranslationToIMUInitial,
                                                           matRotationCorrectionCAMERAtoIMU );
        assert( 0 != pCameraRIGHTwithIMU );
    }

    //ds construct stereo camera (does not throw)
    static void constructCameraSTEREOwithIMU( )
    {
        assert( 0 != pCameraLEFTwithIMU );
        assert( 0 != pCameraRIGHTwithIMU );
        pCameraSTEREOwithIMU = std::make_shared< CStereoCameraIMU >( pCameraLEFTwithIMU, pCameraRIGHTwithIMU );
        assert( 0 != pCameraSTEREOwithIMU );
    }

    //ds simple stereo camera with 3d offsets
    static void constructCameraSTEREO( const Eigen::Vector3d& p_vecTranslationToRIGHT )
    {
        assert( 0 != pCameraLEFT );
        assert( 0 != pCameraRIGHT );
        pCameraSTEREO = std::make_shared< CStereoCamera >( pCameraLEFT, pCameraRIGHT, p_vecTranslationToRIGHT );
        assert( 0 != pCameraSTEREO );
    }

//ds buffered parameters
public:

    static std::shared_ptr< CPinholeCameraIMU > pCameraLEFTwithIMU;
    static std::shared_ptr< CPinholeCameraIMU > pCameraRIGHTwithIMU;
    static std::shared_ptr< CStereoCameraIMU > pCameraSTEREOwithIMU;
    static std::shared_ptr< CPinholeCamera > pCameraLEFT;
    static std::shared_ptr< CPinholeCamera > pCameraRIGHT;
    static std::shared_ptr< CStereoCamera > pCameraSTEREO;

};

//ds allow only single instantiation of parameter base entities (usually in main) LET THIS BE!
std::shared_ptr< CPinholeCameraIMU > CParameterBase::pCameraLEFTwithIMU  = 0;
std::shared_ptr< CPinholeCameraIMU > CParameterBase::pCameraRIGHTwithIMU = 0;
std::shared_ptr< CStereoCameraIMU > CParameterBase::pCameraSTEREOwithIMU = 0;
std::shared_ptr< CPinholeCamera > CParameterBase::pCameraLEFT  = 0;
std::shared_ptr< CPinholeCamera > CParameterBase::pCameraRIGHT = 0;
std::shared_ptr< CStereoCamera > CParameterBase::pCameraSTEREO = 0;

#endif //CPARAMETERBASE_H
