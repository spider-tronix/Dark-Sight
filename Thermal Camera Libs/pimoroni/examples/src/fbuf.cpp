#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <math.h>
#include "headers/MLX90640_API.h"
#include <ncurses.h>
#include <bits/stdc++.h>
#include "lib/fb.h"
#include <sys/select.h>

using namespace std;

#define MLX_I2C_ADDR 0x33

#define IMAGE_SCALE 5

// Valid frame rates are 1, 2, 4, 8, 16, 32 and 64
// The i2c baudrate is set to 1mhz to support these
#define FPS 8
#define FRAME_TIME_MICROS (1000000 / FPS)

// Despite the framerate being ostensibly FPS hz
// The frame is often not ready in time
// This offset is added to the FRAME_TIME_MICROS
// to account for this.
#define OFFSET_MICROS 850


int kbhit(void)
{
struct timeval tv;
fd_set read_fd;

/* Do not wait at all, not even a microsecond */
tv.tv_sec=0;
tv.tv_usec=0;

/* Must be done first to initialize read_fd */
FD_ZERO(&read_fd);

/* Makes select() ask if input is ready:
* 0 is the file descriptor for stdin */
FD_SET(0,&read_fd);

/* The first parameter is the number of the
* largest file descriptor to check + 1. */
if(select(1, &read_fd,NULL, /*No writes*/NULL, /*No exceptions*/&tv) == -1)
return 0; /* An error occured */

/* read_fd now holds a bit map of files that are
* readable. We test the entry for the standard
* input (file 0). */

if(FD_ISSET(0,&read_fd))
/* Character pending on stdin */
return 1;

/* no characters were pending */
return 0;
}


void put_pixel_false_colour(int x, int y, double v, float jpg_image[3][24][32])
{   int xx=x / IMAGE_SCALE;
    int yy=y / IMAGE_SCALE;
    
    // Heatmap code borrowed from: http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
    const int NUM_COLORS = 7;
    static float color[NUM_COLORS][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}};
    int idx1, idx2;
    float fractBetween = 0;
    float vmin = 5.0;
    float vmax = 50.0;
    float vrange = vmax - vmin;
    v -= vmin;
    v /= vrange;
    if (v <= 0)
    {
        idx1 = idx2 = 0;
    }
    else if (v >= 1)
    {
        idx1 = idx2 = NUM_COLORS - 1;
    }
    else
    {
        v *= (NUM_COLORS - 1);
        idx1 = floor(v);
        idx2 = idx1 + 1;
        fractBetween = v - float(idx1);
    }

    int ir, ig, ib;
    ir = (int)((((color[idx2][0] - color[idx1][0]) * fractBetween) + color[idx1][0]) * 255.0);
    ig = (int)((((color[idx2][1] - color[idx1][1]) * fractBetween) + color[idx1][1]) * 255.0);
    ib = (int)((((color[idx2][2] - color[idx1][2]) * fractBetween) + color[idx1][2]) * 255.0);

    for (int px = 0; px < IMAGE_SCALE; px++)
    {
        for (int py = 0; py < IMAGE_SCALE; py++)
        {
            fb_put_pixel(x + px, y + py, ir, ig, ib);
        }
    }

    jpg_image[0][xx][yy] = ir;
    jpg_image[1][xx][yy] = ig;
    jpg_image[2][xx][yy] = ib;
} 

int main()
{
    static uint16_t eeMLX90640[832];
    float emissivity = 1;
    uint16_t frame[834];
    static float image[768];

    float jpg_image[3][24][32];
    float temper_readings[24][32];

    static float mlx90640To[768];
    float eTa;
    static uint16_t data[768 * sizeof(float)];

    auto frame_time = std::chrono::microseconds(FRAME_TIME_MICROS + OFFSET_MICROS);

    MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    switch (FPS)
    {
    case 1:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b001);
        break;
    case 2:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
        break;
    case 4:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b011);
        break;
    case 8:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b100);
        break;
    case 16:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b101);
        break;
    case 32:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b110);
        break;
    case 64:
        MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b111);
        break;
    default:
        printf("Unsupported framerate: %d", FPS);
        return 1;
    }
    MLX90640_SetChessMode(MLX_I2C_ADDR);

    paramsMLX90640 mlx90640;
    MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    fb_init();

    while (!kbhit())
    {   
        auto start = std::chrono::system_clock::now();
        auto error = MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
        if (error < 0)
        {
            printf("Failed to get frame data.\n");
            exit(1);
        }
        // MLX90640_InterpolateOutliers(frame, eeMLX90640);

        eTa = MLX90640_GetTa(frame, &mlx90640);
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);

        MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, 1, &mlx90640);
        MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, 1, &mlx90640);
        for (int yy = 0; yy < 24; yy++)
        {
                
            for (int xx = 0; xx < 32; xx++)
            {
                float val = mlx90640To[32 * (23 - yy) + xx];
                temper_readings[yy][xx] = val;
                put_pixel_false_colour((yy * IMAGE_SCALE), (xx * IMAGE_SCALE), val, jpg_image);
            }
        }
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::this_thread::sleep_for(std::chrono::microseconds(frame_time - elapsed));
    }
    string filename;
    ifstream new_file;

    new_file.open("/home/pi/sambaishere/timestamp.txt", ios::in);
    if(!new_file)
    {
        cout<<"no file\n";
        return 0;
    }
    // new_file >> filename;
    getline(new_file, filename);

    new_file.close();

    ofstream out_file;
    out_file.open("/home/pi/sambaishere/" + filename + "/temp.txt", ios::out);
    cout<<"Writing to file "<<"/home/pi/sambaishere/" + filename + "/temp.txt"<<endl;
    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 32; ++j)
            out_file<<temper_readings[i][j] << '\t';
        out_file<< '\n';
    }

    out_file.close();
    out_file.open("/home/pi/sambaishere/" + filename + "/jpg_temp.txt", ios::out);
    cout<<"Writing to file "<<"/home/pi/sambaishere/" + filename + "/jpg_temp.txt"<<endl;
    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < 24; ++i)
        {
            for (int j = 0; j < 32; ++j)
                out_file << int(jpg_image[c][i][j]) << '\t';
            out_file << '\n';
        }
        out_file << "\n#\n";
    }

    out_file.close();

    fb_cleanup();
    return 0;
}
