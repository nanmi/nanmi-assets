#include "Python.h"
#include <iostream>

int main(int argc, char const *argv[])
{
        //C++调用python启动gst-rtsp-server
    Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化
    if (!Py_IsInitialized())
    {
        printf("初始化失败！");
        return 0;
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../lib/')");//这一步指向生成的so包路径即可

    // 载入名为your_file的脚本
    PyObject* pModule = PyImport_ImportModule("script_function");
    if (!pModule) {
        printf("can't find your_file.py");
        return -1;
    }

    PyObject* pFunc = PyObject_GetAttrString(pModule, "add");

    /*PyObject* Py_BuildValue(char *format, ...)
    把C++的变量转换成一个Python对象。当需要从
    C++传递变量到Python时，就会使用这个函数。此函数
    有点类似C的printf，但格式不同。常用的格式有
    s 表示字符串，
    i 表示整型变量，
    f 表示浮点数，
    O 表示一个Python对象。
    这里我要传的是字符串所以用s，注意字符串需要双引号！
    */
    PyObject* pArgs = Py_BuildValue("ii", 2, 3);
    // 调用Python函数
    PyObject* pRet = PyObject_CallObject(pFunc, pArgs);
    float res = 0;
    PyArg_Parse(pRet, "f", &res);//转换返回类型
    printf(">>>>>>>> %f \n",res);
    // 关闭Python
    Py_Finalize();
    return 0;
}
