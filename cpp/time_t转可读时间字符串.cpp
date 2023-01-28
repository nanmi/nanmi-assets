#include <time.h>
#include <stdio.h>

void time_t2_time(time_t timetTime)
{
    char szTime[24] = {0};
    struct tm *pTmTime;

    //time_t 结构转换成tm结构
    pTmTime = localtime(&timetTime);
 
    //验证tm类型数据是否正确
    snprintf(szTime, sizeof(szTime)-1,
     "%d-%02d-%02d-%02d-%02d-%02d",
        pTmTime->tm_year+1900,
        pTmTime->tm_mon+1,
        pTmTime->tm_mday,
        pTmTime->tm_hour,
        pTmTime->tm_min,
        pTmTime->tm_sec);
    
    printf("szTime = %s\n", szTime);
}

int main()
{
    time_t timetTime = time(NULL);
    printf("timetTime = %ld\n", timetTime);
 
    time_t2_time(timetTime);
    
    return 0;

}