import os

# 定义 IP 配置
ip_config = """
157.240.20.8 github.global.ssl.fastly.net
20.205.243.166 github.com
"""

# 设置 /etc/hosts 文件路径
hosts_file = "/etc/hosts"

# 将 IP 配置追加到 hosts 文件
def add_to_hosts():
    try:
        # 打开 hosts 文件并追加内容
        with open(hosts_file, 'a') as f:
            f.write(ip_config)
        print("IP 配置已成功写入 /etc/hosts 文件。")
    except PermissionError:
        print("没有权限写入 /etc/hosts 文件，请确保以管理员权限运行脚本。")
        return False
    return True

# 刷新 DNS 缓存
def refresh_dns():
    try:
        # 使用系统命令重启网络服务来刷新 DNS 缓存
        os.system("sudo /etc/init.d/networking restart")
        print("DNS 缓存已成功刷新。")
    except Exception as e:
        print(f"刷新 DNS 缓存失败: {e}")

# 主执行流程
def main():
    if add_to_hosts():
        refresh_dns()

if __name__ == "__main__":
    main()
