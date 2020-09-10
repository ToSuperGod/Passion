#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#define MSGKEY 97

struct msgform
{
    long mtype;
    char mtex[1024];
}msg;

int msgqid;

void server()
{
    msgqid = msgget(MSGKEY,0777|IPC_CREAT);  /*消息队列名字，不存在创建，存在返回*/
    do{
        msgrcv(msgqid,&msg,1030,0,0);  
        sleep(1);
        printf("server received 1K information\n");
    }while(msg.mtype!=1);
    msgctl(msgqid,IPC_RMID,0);   /*删除消息队列*/
    exit(0);
}
void main()
{
    server();
}
