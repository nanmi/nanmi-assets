
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int file_write(std::string file_path, std::string content)
{
    std::ofstream fout(file_path);
    if(fout.fail()){
        std::cout << "open file error" << std::endl;
        return -1;
    }
    
    fout << content;

    fout.close();
    return 0;
}


int file_read(std::string file_path, std::vector<std::string>& content)
{
	std::fstream fin(file_path);   //读取文件
	if (fin.fail())
	{
		std::cout << "open file error" << std::endl;
		return -1;
	}

    std::string line_content;
    while (getline(fin, line_content))
    {
        content.emplace_back(line_content);
        std::cout << "a = " << line_content << std::endl;
    }

	fin.close();       //关闭文件
	
    return 0;
}


int main() {
    // write
    file_write("./aaaaa.txt", "11111,\n22222\n");

    // read
    std::vector<std::string> con;
    file_read("./aaaaa.txt", con);
    for (size_t i = 0; i < con.size(); i++)
    {
        /* code */
        std::cout << "b = " << con[i] << std::endl;
    }

    return 0;
}
