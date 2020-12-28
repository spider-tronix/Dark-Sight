/*

To be build using Pimoroni 'make' configs.

Program to be run on RPi.(Triggered by 'main.py')

Uses TCP socket to send thermal readings.

*/
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>

#define BUFLEN 1024 // Size of array holding msg to be sent

#define SRV_PORT 1234

#define SRV_IP "192.168.0.104"

#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include "headers/MLX90640_API.h"

#define ANSI_COLOR_RED ""
#define ANSI_COLOR_GREEN ""
#define ANSI_COLOR_YELLOW ""
#define ANSI_COLOR_BLUE ""
#define ANSI_COLOR_MAGENTA ""
#define ANSI_COLOR_CYAN ""
#define ANSI_COLOR_NONE ""
#define ANSI_COLOR_RESET ""

#define FMT_STRING "%+06.2f "
// #define FMT_STRING "\u2588\u2588"

#define MLX_I2C_ADDR 0x33

// socket stuff
int socket_desc;
struct sockaddr_in server;
char *message;

void socket_init()
{
    //Create socket
    socket_desc = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_desc == -1)
    {
        printf("Could not create socket");
    }

    server.sin_addr.s_addr = inet_addr(SRV_IP);
    server.sin_family = AF_INET;
    server.sin_port = htons(SRV_PORT);

    //Connect to remote server
    puts("\nConnecting to server....");
    while (connect(socket_desc, (struct sockaddr *)&server, sizeof(server)) < 0)
        ; // Wait till connection is established

    puts("Connected\n");
}
void socket_send(char buf[BUFLEN])
{
    if (send(socket_desc, buf, strlen(buf), 0) < 0)
    {
        puts("Send failed");
    }
}

int main()
{
    char buf[BUFLEN];

    socket_init();
    // int iter = 0;

    int state = 0;
    printf("Starting...\n");
    static uint16_t eeMLX90640[832];
    float emissivity = 1;
    uint16_t frame[834];
    static float image[768];
    float eTa;
    static uint16_t data[768 * sizeof(float)];

    std::fstream fs;

    MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0x06); // Set refresh rate here
    MLX90640_SetChessMode(MLX_I2C_ADDR);
    //MLX90640_SetSubPage(MLX_I2C_ADDR, 0);
    printf("Configured...\n");

    paramsMLX90640 mlx90640;
    MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);

    int refresh = MLX90640_GetRefreshRate(MLX_I2C_ADDR);
    printf("EE Dumped...\n");

    int frames = 30;
    int subpage;
    static float mlx90640To[768];
    while (1)
    {
        state = !state;
        //printf("State: %d \n", state);
        MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
        MLX90640_InterpolateOutliers(frame, eeMLX90640);
        eTa = MLX90640_GetTa(frame, &mlx90640);
        subpage = MLX90640_GetSubPageNumber(frame);
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);

        MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, 1, &mlx90640);
        MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, 1, &mlx90640);

        sprintf(buf, "Subpage: %d\n", subpage);
        printf("%s", buf); // !!Original
        socket_send(buf);
        // MLX90640_SetSubPage(MLX_I2C_ADDR,!subpage);

        for (int x = 0; x < 32; x++)
        {
            for (int y = 0; y < 24; y++)
            {
                float val = mlx90640To[32 * (23 - y) + x];

                if (val > 99.99)
                    val = 99.99;

                sprintf(buf, FMT_STRING ANSI_COLOR_RESET, val);
                socket_send(buf); // !! Send through socket
            }
            socket_send((char *)"\n");
        }
        socket_send((char *)"End");
        socket_send((char *)"\x1b[33A");
    }
    return 0;
}
