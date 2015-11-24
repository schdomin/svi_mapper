#ifndef CEXCEPTIONDETECTIONFAILED_H_
#define CEXCEPTIONDETECTIONFAILED_H_

#include <exception>

class CExceptionEpipolarLine: public std::exception
{

public:

    CExceptionEpipolarLine( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionEpipolarLine( )
    {

    }

private:

    const std::string m_strExceptionDescription;

public:

    virtual const char* what( ) const throw( )
    {
        return m_strExceptionDescription.c_str( );
    }
};

#endif //CEXCEPTIONDETECTIONFAILED_H_
