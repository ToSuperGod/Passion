#include <stdlib.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <stdio.h>
#include <unistd.h>
#define MSGKEY 97
struct msgform
{
    long mtype;
    char mtex[1024];
}msg;

int msgqid;

void client()
{
    int i;
    msgqid = msgget(MSGKEY,0777);  /*消息队列标识码*/
    for(i=3;i>=1;i--){
        msg.mtype=i;  /*指定发送消息的类型*/
        sleep(1);
        printf("client send 1K information\n");
        msgsnd(msgqid,&msg,1024,0);  /*标识码、指向准备发送的消息、消息的长度*/
    }
    exit(0);
}
void main()
{
    client();
}
/*消息队列发送的消息有类型*/
