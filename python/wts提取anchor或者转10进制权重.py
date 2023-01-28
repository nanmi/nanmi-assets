import struct


def intToHex(intValue):  # int转16进制
    return hex(intValue)[2:].upper()
 
def hexToInt(hexString):  # 16进制转int
    return int(hexString, 16)
 
def floatToHex(floatValue):  # float转16进制
    return struct.pack('>f', floatValue).hex().upper()
 
def hexToFloat(hexString):  # 16进制转float
    return struct.unpack('>f', bytes.fromhex(hexString))[0]
 
def doubleToHex(doubleValue):  # double转16进制
    return struct.pack('>d', doubleValue).hex().upper()
 
def hexToDouble(hexString):  # 16进制转double
    return struct.unpack('>d', bytes.fromhex(hexString))[0]

def wts_bytes2float(wts_file, wts_file_new, only_extract_anchor=True) -> None:
    with open(wts_file, 'r') as f:
        with open(wts_file_new, 'w+') as f1:
            while True:
                line_data = f.readline()
                if line_data is None or line_data == '':
                    break
                raw_data = line_data.split(' ')
                
                # 去除空格
                for item in raw_data:
                    item.strip()
                    if item == '' or item == ' ':
                        raw_data.remove(item)

                key = raw_data[0]
                if only_extract_anchor:
                    if key.split('.')[-1] == 'anchor_grid':
                        f1.write(f'{key} \n')
                        for raw in raw_data[2:]:
                            raw.strip('\n')
                            f1.write(f'{int(hexToFloat(raw))} ')
                else:
                    # raw_data_length = raw_data[1]
                    f1.write(f'{key} \n')
                    for raw in raw_data[2:]:
                        raw.strip('\n')
                        f1.write(f'{hexToFloat(raw)} ')
                    f1.write("\n==========================\n")


if __name__ == '__main__':
    wts_file = './test.wts'
    wts_file_new = 'test.txt'
    wts_bytes2float(wts_file, wts_file_new, only_extract_anchor=True)
