#include<stdio.h>
#include<sys/types.h>
#include<unistd.h>
#include<sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <stdlib.h>
int wait_flag=1;
void stop(){

}
void stop2(){
    wait_flag = 0;
}
int main(){

    pid_t pid1,pid2;
    signal(SIGINT,stop);  /*键盘中断*/
    pid1=fork(); /*返回*/


    if(pid1>0){
        pid2=fork();
        if(pid2>0){
            sleep(10);
            kill(pid1,20);
            wait(0);
            kill(pid2,12);
            wait(0);
            printf("\nParent process is killed!\n");
            exit(0);
        }else{
            signal(12,stop2);
            while(wait_flag)
                ;
            printf("Child Processl2 is killed by parent!\n");
            exit(0);
        }
    }else if(pid1==0){
        signal(20,stop2);
        while(wait_flag)
            ;
        printf("Child Processl1 is killed by parent!\n");
        exit(0);
    }
}
