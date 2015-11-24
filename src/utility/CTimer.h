#ifndef CTIMER_H
#define CTIMER_H

#include <iostream>
#include <chrono>
#include <string>



class CTimer
{

public:

    static const std::string getTimestampString( )
    {
        //ds current time
        const std::time_t tmCurrent = std::time( NULL );

        //ds compute stamp and return
        char chBufferTimestamp[100];
        std::strftime( chBufferTimestamp, sizeof( chBufferTimestamp ), "%Y-%m-%d-%H%M%S", std::localtime( &tmCurrent ) );
        return chBufferTimestamp;
    }

    static const double getTimeSeconds( )
    {
        return std::chrono::system_clock::now( ).time_since_epoch( ).count( )/1e9;
    }

};

#endif //CTIMER_H
