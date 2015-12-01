#ifndef TYPESTHREADING_H
#define TYPESTHREADING_H

#include <mutex>
#include <condition_variable>
#include <vector>

#include "types/CKeyFrame.h"



struct CHandleThreadMapping
{
    CHandleThreadMapping( ): vecKeyFramesToAdd( std::vector< CKeyFrame* >( 0 ) )
    {
        //ds nothing to do
    }

    //ds communication
    std::mutex cMutex;
    std::condition_variable cConditionVariable;
    bool bActive               = true;
    bool bTerminationRequested = false;

    //ds data
    std::vector< CKeyFrame* > vecKeyFramesToAdd;
};

#endif //TYPESTHREADING_H
