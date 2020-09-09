// findPosion.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <fstream>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include <vector>
#include <ctime>
using namespace std;
using namespace cv;

//异常处理都没有写
//尽可能的所有数组均使用 vector
void above()
{
	Mat im = imread("D:/study/picture/TPs5.PNG");
	namedWindow("DT");
	imshow("photo", im);
	waitKey(1000);
	destroyWindow("photo");  // 查看图片
	if (im.empty())
	{
		cout << "图片读取错误，请检查" << endl;
		exit(1);
	}

	int pixelR, pixelG, pixelB;
	for (int r = 0; r < im.rows; r++)
	{
		for (int c = 0; c < im.cols; c++)
		{
			pixelB = im.at<Vec3b>(r, c)[0];
			pixelG = im.at<Vec3b>(r, c)[1];
			pixelR = im.at<Vec3b>(r, c)[2];
			/*if (pixelB == 0 && pixelG == 0 && pixelR == 0) {
				//im.at<Vec3b>(r, c)[0] = 0;
				//im.at<Vec3b>(r, c)[1] = 0;
				//im.at<Vec3b>(r, c)[2] = 0;
				cout << r << "行 " << c << "列" << endl;
			}*/
			cout << r << "行," << c << "列:" << pixelR << " " << pixelG << " " << pixelB << "  ";
		}
		cout << endl;
	}
	namedWindow("DT");
	imshow("photo", im);
	waitKey(0);

}



void solveCore(int num_cir, vector <int> fouces_X, vector <int> fouces_Y, int plot_num, Mat img_f, int check_start)
{
	int number;
	for (int i = check_start; i < num_cir - 1; i++) {
		number = sqrt((fouces_X[check_start] - fouces_X[i + 1]) * (fouces_Y[check_start] - fouces_Y[i + 1]));
		if (number < plot_num && number > 2) {
			int c = (fouces_Y[check_start] + fouces_Y[i + 1]) / 2;
			int r = (fouces_X[check_start] + fouces_X[i + 1]) / 2;
			cv::circle(img_f, Point(c, r), 5, cv::Scalar(0, 0, 255), -1);
		}
	}
}


void solveCore_t(vector <int> center_X, vector <int> center_Y, int max_num, int check_center, int plot_num, Mat img_f)
{
	int number;
	int check = 0;
	for (int i = check_center; i < max_num - 1; i++) {
		number = sqrt((center_X[check_center] - center_X[i + 1]) * (center_Y[check_center] - center_Y[i + 1]));
		if (number < plot_num * 2) {
			int c = (center_X[check_center] + center_X[i + 1]) / 2;
			int r = (center_Y[check_center] + center_Y[i + 1]) / 2;
			cv::circle(img_f, Point(c, r), 5, cv::Scalar(0, 0, 255), -1);
			check++;
		}
	}
	if (check == 0) {
		cv::circle(img_f, Point(center_X[check_center], center_Y[check_center]), 5, cv::Scalar(0, 0, 255), -1);
	}
}
int Read()
{
	//读取图片的长度和宽度
	Mat img = imread("1.JPG");
	namedWindow("photo");
	imshow("photo", img);
	waitKey(3000);
	//cvReleaseImage(&img);
	//cvDestroyWindow(strWindowname);
	destroyWindow("photo"); //关闭窗口，并取消之前分配的与窗口相关的所有内存空间
	if (img.empty())
	{
		cout << "图片读取错误，请检查" << endl;
		exit(1);
	}
	//int pixelR, pixelG, pixelB;//像素rgb的值
	cout << "此图片像素贞一共" << img.rows << "行，" << img.cols << "列" << endl;

	//计算比例尺,数据还需精确
	int tlong;//实际长度和宽度
	int plotting_scale;//比例尺
	int num_1 = img.cols;  //行像素点
	cout << "请输入该地图在实际中的长度 tlong（单位km） :";
	cin >> tlong;
	if (tlong < 3)
		return 0;
	tlong *= 1000; //转化成米
	plotting_scale = tlong / num_1;
	cout << "该图片的比例尺为1：" << plotting_scale << endl;

	//图片分块
	int row, col, num_row, num_col;
	//int num_img_cols = img.cols;
	//int num_img_rows = img.rows;
	cout << "你想将图片均分成几行row几列col,请输入行和列的值: ";
	cin >> row >> col;
	num_row = img.rows / row; //行宽度
	num_col = img.cols / col;//列宽度
	for (int i = 1; i < row; i++) {
		cv::Point start = cv::Point(1, i * num_row);
		cv::Point end = cv::Point(img.cols, i * num_row);
		cv::line(img, start, end, cv::Scalar(0, 0, 255));
	}
	for (int i = 1; i < col; i++) {
		cv::Point start = cv::Point(i * num_col, 1);
		cv::Point end = cv::Point(i * num_col, img.rows);
		cv::line(img, start, end, cv::Scalar(0, 0, 255));
	}
	namedWindow("photo_1");
	imshow("photo_1", img);
	waitKey(1000);
	destroyWindow("photo_1");


	//输入各块的人流量 ,提示点一个一个显示
	//row行  col 列
	cout << "请输入图中标记点的人流量和车辆量" << endl;
	int people[20][20], cars[20][20];//人流量和车流量数组
	int center_point_X[20][20], center_point_Y[20][20];//记录中心点坐标数组
	for (int i = 1; i <= col; i++) {
		for (int j = 1; j <= row; j++) {
			cv::Point point;
			point.x = (i - 1) * num_col + num_col / 2;
			point.y = (j - 1) * num_row + num_row / 2;
			center_point_Y[i][j] = (i - 1) * num_row + num_row / 2;
			center_point_X[i][j] = (j - 1) * num_col + num_col / 2;
			cout << "中心坐标：" << center_point_X[i][j] << " " << center_point_Y[i][j] << endl;
			cv::circle(img, point, 5, cv::Scalar(0, 0, 255),-1);
			/*namedWindow("photo");
			imshow("photo", img);
			waitKey(10);
			destroyWindow("photo");*/
			//cin >> people[i][j];//从1开始的
			//cin >> cars[i][j];//从1开始的
			srand((int)time(NULL));// 随机数
			people[i][j] = rand() % 1000;
			cars[i][j] = rand() % 1000;
			cout << cars[i][j] << endl;
		}
	}
	namedWindow("photo");
	imshow("photo", img);
	waitKey(1000);
	destroyWindow("photo");


	//方案二核心算法
	//row行  col 列
	//10选1
	//车流量取最大值算法
	int max_num = 0; //10选1
	if (row * col < 10) {
		max_num = 1;
	}
	else {
		max_num = row* col / 10;
		if (row * col % 10 > 5) //16个块选两个点
			max_num++;
	}
	int max_arry[20] = { 0 };//存储车流量最大值
	int chack_x[20] = { 0 };
	int chack_y[20] = { 0 };// 标记最大点xy（行、列）值
	int max_cars;//最大车流量
	//如果个点车流量相等则看人流量
	for (int r = 0; r < max_num; r++) {
		max_cars = 0;
		for (int i = 1; i <= row; i++) {
			for (int j = 1; j <= col; j++) {
				if (max_cars < cars[i][j]) {
					if (r == 0) {
						max_cars = cars[i][j];
						chack_x[r] = i;
						chack_y[r] = j;
						max_arry[r] = max_cars;
					}
					else {
						int chack_s = 0;
						for (int s = 0; s < r; s++) {
							if (chack_x[s] == i && chack_y[s] == j)
								chack_s++;
						}
						if (chack_s == 0) {
							max_cars = cars[i][j];
							chack_x[r] = i;
							chack_y[r] = j;
							max_arry[r] = max_cars;
						}
					}
				}
			}
		}
	}
	for (int i = 0; i < max_num; i++) {
		cout << chack_x[i] << " " << chack_y[i] << endl;
	}





	//画圈算法
	//比例尺那里还需要优化细节
	//据中心点2公里处寻找建立车库位置
	int distance, plot_num = 0; //距离和像素点个数
	cout << "请输入您想要距离车辆密集处多远建立车库(单位：km)：";
	cin >> distance;
	plot_num = distance * 1000 / plotting_scale;  //这个为比例尺
	cout << "像素点个数：" << plot_num << endl;
	Mat img_t = imread("1.JPG");  //重新打开图片
	vector <int> center_X;//存储圆的中心坐标
	vector <int> center_Y;
	for (int i = 0; i < max_num; i++) {  
		int r = chack_x[i];
		int c = chack_y[i];
		cv::circle(img_t, Point(center_point_X[r][c], center_point_Y[r][c]), plot_num, cv::Scalar(0, 0, 0));//画圆可以填充
		center_X.push_back(center_point_X[r][c]);
		center_Y.push_back(center_point_Y[r][c]);
		cout << center_point_X[r][c] << "一个圆中心坐标 " << center_point_Y[r][c] << endl;  
	}
	namedWindow("photo");
	imshow("photo", img_t);
	waitKey(0);
	destroyWindow("photo");



	//提取圆上各点坐标,并把图像二值化
	int rgB, rGb, Rgb; //rgb像素值
	for (int r = 0; r < img_t.rows; r++)
	{
		for (int c = 0; c < img_t.cols; c++)
		{
			rgB = img_t.at<Vec3b>(r, c)[0];
			rGb = img_t.at<Vec3b>(r, c)[1];
			Rgb = img_t.at<Vec3b>(r, c)[2];
			if (rgB != 0 && rGb != 0 && Rgb != 0) {
				img_t.at<Vec3b>(r, c)[0] = 255;
				img_t.at<Vec3b>(r, c)[1] = 255;
				img_t.at<Vec3b>(r, c)[2] = 255;
				//cout << r << "行 " << c << "列" << endl;//画圆点
			}
		}
	}


	//找交点,并标记
	int num_cir = 0;
	vector <int> fouces_X;  //记录焦点坐标 列
	vector <int> fouces_Y; // 记录焦点坐标 行
	for (int r = 0; r < img_t.rows; r++)
	{
		for (int c = 0; c < img_t.cols; c++)
		{
			rgB = img_t.at<Vec3b>(r, c)[0];
			rGb = img_t.at<Vec3b>(r, c)[1];
			Rgb = img_t.at<Vec3b>(r, c)[2];
			if (rgB == 0 && rGb == 0 && Rgb == 0) {
				if (r != 0 && c != 0 && r < img.rows - 1 && c < img.cols - 1) {
					int biu_r[] = { r - 1,r + 1,r,r,r - 1,r - 1,r + 1,r + 1 };//黑色点周围的8个点
					int bui_c[] = { c,c,c - 1,c + 1,c - 1,c + 1,c - 1,c + 1 };
					int num_black = 0;
					for (int i = 0; i < 8; i++) {
						int num_br = biu_r[i];
						int num_bc = bui_c[i];
						rgB = img_t.at<Vec3b>(num_br, num_bc)[0];
						rGb = img_t.at<Vec3b>(num_br, num_bc)[1];
						Rgb = img_t.at<Vec3b>(num_br, num_bc)[2];//读取黑色点周围个点的像素值
						if (rgB == 0 && rGb == 0 && Rgb == 0) {
							num_black++;
						}
					}
					if (num_black > 2) {
						//cv::circle(img_t, Point(c, r), 2, cv::Scalar(0, 0, 255));  //在标记点画圆
						fouces_Y.push_back(c);
						fouces_X.push_back(r);//记录坐标
						num_cir++; //记录标记点个数
						cout << "标记点个数" << num_cir << endl;
						cout << c << " " << r << endl;
					}
				}
			}
			
		}
	}
	namedWindow("photo");
	imshow("photo", img_t);
	waitKey(0);
	destroyWindow("photo");


	//确定车库位置
	//找最近的标记点 ，半径范围内
	//没有标记点，直接取圆心
	Mat img_f = imread("1.JPG");
	if (num_cir == 0) {
		for (int i = 0; i < max_num; i++) {
			int r = chack_x[i];
			int c = chack_y[i];
			cv::circle(img_f, Point(center_point_X[r][c], center_point_Y[r][c]), 5, cv::Scalar(0, 0, 255),-1);//画圆可以填充
			
		}
		cout << "合适的建立车库位置已经标出" << endl;//有待优化
		namedWindow("photo");
		imshow("photo", img_f);
		waitKey(0);
		destroyWindow("photo");
	}
	else//有焦点的解决方案
	{
		cout << "合适的建立车库位置已经标出" << endl;//有待优化
		int check_start = 0;
		solveCore(num_cir,fouces_X,fouces_Y,plot_num,img_f,check_start);
		while (check_start < num_cir - 1) {//细看
			check_start++;
			solveCore(num_cir, fouces_X, fouces_Y, plot_num, img_f, check_start);
		}

		int check_center = 0;
		solveCore_t(center_X, center_Y,max_num,check_center,plot_num,img_f);  //方案二
		while (check_center < max_num - 1) {
			check_center++;
			solveCore_t(center_X, center_Y, max_num, check_center,plot_num,img_f);
		}

		namedWindow("photo");
		imshow("photo", img_f);
		waitKey(0);
		destroyWindow("photo");

	}

	return 0;
}






int main()
{

	// 测试代码，训练使用
	//above();


	//前一段代码
	Read();
	/*ofstream output("pixelValue.txt");//没有此文件则重新创建，在项目中
	output << "此图片一共" << img.rows << "行，" << img.cols << "列,三个值得顺序分别为R,G,B的值" << endl;
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			pixelB = img.at<Vec3b>(r, c)[0];
			pixelG = img.at<Vec3b>(r, c)[1];
			pixelR = img.at<Vec3b>(r, c)[2];
			output << r << "行," << c << "列:" << pixelR << " " << pixelG << " " << pixelB << "  ";
		}
		output << endl << endl;
	}*/
	return 0;
}
