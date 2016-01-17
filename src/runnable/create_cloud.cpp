#include "utility/CLogger.h"



int32_t main( int32_t argc, char **argv )
{
    //ds parameters
    const uint64_t uNumberOfPoints              = 100;
    const uint64_t uNumberOfDescriptorsPerPoint = 100;
    const uint32_t uDescriptorSizeBits          = 256;

    CLogger::openBox( );
    std::printf( "(main) uNumberOfPoints              := %lu\n", uNumberOfPoints );
    std::printf( "(main) uNumberOfDescriptorsPerPoint := %lu\n", uNumberOfDescriptorsPerPoint );
    std::printf( "(main) uDescriptorSizeBits          := %u\n", uDescriptorSizeBits );
    CLogger::closeBox( );

    //ds probability core
    std::random_device cRandomSource;
    std::mt19937 cGenerator( cRandomSource( ) );
    std::uniform_int_distribution< int8_t > cDistribution( 0, 1 );

    //ds construct filestring and open dump file
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "clouds/artifical_0_%03u_%lu.cloud", uDescriptorSizeBits, uNumberOfPoints*uNumberOfDescriptorsPerPoint );
    std::ofstream ofCloud( chBuffer, std::ofstream::out );
    assert( ofCloud.is_open( ) );
    assert( ofCloud.good( ) );

    //ds write bits for all points
    for( uint64_t u = 0; u < uNumberOfPoints; ++u )
    {
        for( uint64_t v = 0; v < uNumberOfDescriptorsPerPoint; ++v )
        {
            for( uint32_t w = 0; w < uDescriptorSizeBits; ++w )
            {
                CLogger::writeDatum( ofCloud, cDistribution( cGenerator ) );
            }
        }
    }

    //ds close file
    ofCloud.close( );
    std::fflush( stdout);
    return 0;
}
