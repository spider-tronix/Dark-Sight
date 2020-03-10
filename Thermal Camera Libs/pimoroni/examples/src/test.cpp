#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include "headers/MLX90640_API.h"

#define ANSI_COLOR_RED     ""
#define ANSI_COLOR_GREEN   ""
#define ANSI_COLOR_YELLOW  ""
#define ANSI_COLOR_BLUE    ""
#define ANSI_COLOR_MAGENTA ""
#define ANSI_COLOR_CYAN    ""
#define ANSI_COLOR_NONE    ""
#define ANSI_COLOR_RESET   ""

#define FMT_STRING "%+06.2f "
// #define FMT_STRING "\u2588\u2588"

#define MLX_I2C_ADDR 0x33

int main(){
    int state = 0;
    printf("Starting...\n");
    static uint16_t eeMLX90640[832];
    float emissivity = 1;
    uint16_t frame[834];
    static float image[768];
    float eTa;
    static uint16_t data[768*sizeof(float)];

    std::fstream fs;

    MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
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
    while (1){
        state = !state;
        //printf("State: %d \n", state);
        MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
        MLX90640_InterpolateOutliers(frame, eeMLX90640);
        eTa = MLX90640_GetTa(frame, &mlx90640);
        subpage = MLX90640_GetSubPageNumber(frame);
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);

        MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, 1, &mlx90640);
        MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, 1, &mlx90640);

        printf("Subpage: %d\n", subpage);
        // MLX90640_SetSubPage(MLX_I2C_ADDR,!subpage);

        for(int x = 0; x < 32; x++){
            for(int y = 0; y < 24; y++){
                //std::cout << image[32 * y + x] << ",";
                float val = mlx90640To[32 * (23-y) + x];



                if(val > 99.99) val = 99.99;
                if(val > 32.0){
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
                else if(val > 29.0){
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
                else if (val > 26.0){
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
                else if ( val > 20.0 ){
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
                else if (val > 17.0) {
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
                else if (val > 10.0) {
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
                else {
                    printf(FMT_STRING ANSI_COLOR_RESET, val) ;
                }
            }
            std::cout << std::endl;
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(20));
	printf("End");        
	printf("\x1b[33A");
    }
    return 0;
}
