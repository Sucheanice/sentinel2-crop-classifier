# RAGFlow 迁移部署说明

本目录是从旧服务器（10.62.11.15）整体导出/备份的 **RAGFlow v0.17.2** 完整数据与配置，用于在新服务器上恢复一个完全一致的 RAGFlow 实例。

## 目录内容

| 文件 | 说明 |
|------|------|
| `ragflow_src.tar.gz` | RAGFlow 源码 + 配置 + 自定义代码（解压后得到 `ragflow/` 目录） |
| `docker_mysql_data.tar.gz` | MySQL 元数据卷（用户、知识库、对话记录） |
| `docker_esdata01.tar.gz` | Elasticsearch 检索/向量索引卷 |
| `docker_minio_data.tar.gz` | MinIO 文档文件卷 |
| `docker_redis_data.tar.gz` | Redis 缓存卷 |

## 一、前置条件

新服务器需安装：

- Docker（建议 24+，原机为 29.1.5）
- Docker Compose v2（原机为 v5.0.1，即 `docker compose` 子命令）

确认命令：

```bash
docker --version
docker compose version
```

## 二、部署步骤

> 以下命令均在当前 `ragflow_migrate` 目录下执行。

### 1. 解压源码

```bash
mkdir -p /opt/ragflow_deploy
tar -xzf ragflow_src.tar.gz -C /opt/ragflow_deploy
```

解压后得到 `/opt/ragflow_deploy/ragflow/`，内含 `docker/`（compose 文件）、`conf/`、`history_data_agent/` 等。

### 2. 拉取镜像（需联网）

```bash
docker pull infiniflow/ragflow:v0.17.2
docker pull mysql:8.0.39
docker pull elasticsearch:8.11.3
docker pull quay.io/minio/minio:RELEASE.2023-12-20T01-00-02Z
docker pull valkey/valkey:8
```

> 若新服务器无法联网，请改用文末「离线迁移」方式。

### 3. 创建数据卷并恢复数据

```bash
docker volume create docker_mysql_data
docker volume create docker_esdata01
docker volume create docker_minio_data
docker volume create docker_redis_data

docker run --rm -v docker_mysql_data:/data -v "$(pwd)":/backup alpine \
  sh -c "tar -xzf /backup/docker_mysql_data.tar.gz -C /data"

docker run --rm -v docker_esdata01:/data -v "$(pwd)":/backup alpine \
  sh -c "tar -xzf /backup/docker_esdata01.tar.gz -C /data"

docker run --rm -v docker_minio_data:/data -v "$(pwd)":/backup alpine \
  sh -c "tar -xzf /backup/docker_minio_data.tar.gz -C /data"

docker run --rm -v docker_redis_data:/data -v "$(pwd)":/backup alpine \
  sh -c "tar -xzf /backup/docker_redis_data.tar.gz -C /data"
```

> 上面用临时 alpine 容器解包，避免直接操作 `/var/lib/docker`。若新机没有 alpine 镜像，先执行 `docker pull alpine`。

### 4. 启动服务

```bash
cd /opt/ragflow_deploy/ragflow/docker
docker compose -p docker up -d
```

> `-p docker` 强制项目名为 `docker`，确保卷名 `docker_mysql_data` 等与上面手动创建的卷一一对应。

### 5. 验证

```bash
docker compose -p docker ps
curl -s -o /dev/null -w 'API HTTP %{http_code}\n' http://localhost:9380/
curl -s -o /dev/null -w 'Web HTTP %{http_code}\n' http://localhost:8080/
```

访问地址：

- Web 界面：`http://<服务器IP>:8080`
- API 服务：`http://<服务器IP>:9380`

## 三、端口映射

| 服务 | 宿主机端口 | 容器端口 |
|------|-----------|---------|
| RAGFlow Web | 8080 | 80 |
| RAGFlow API | 9380 | 9380 |
| RAGFlow HTTPS | 8443 | 443 |
| Elasticsearch | 1200 | 9200 |
| MySQL | 5455 | 3306 |
| MinIO API | 9000 | 9000 |
| MinIO 控制台 | 9001 | 9001 |
| Redis | 6379 | 6379 |

如新服务器端口冲突，修改 `ragflow/docker/.env` 中对应的 `*_PORT` 变量，然后重新 `docker compose -p docker up -d` 即可。

## 四、重要注意事项

1. **自定义代码**：`ragflow/docker/doc.py` 是自定义的 SDK 接口，已通过 `docker-compose.yml` 挂载覆盖镜像内文件，迁移后自动生效，无需额外处理。
2. **默认密码**：MySQL / Elasticsearch / MinIO / Redis 密码均为 `infini_rag_flow`（见 `ragflow/docker/.env`）。
3. **卷名前缀**：数据卷名固定为 `docker_` 前缀（compose 项目名为 `docker`），恢复时务必使用上面指定的卷名。
4. **时区**：`TIMEZONE=Asia/Shanghai`（在 `.env` 中）。
5. **文档引擎**：使用 Elasticsearch（`DOC_ENGINE=elasticsearch`），请勿改动，否则需同时迁移 infinity 卷。

## 五、离线迁移（无外网时）

若新服务器无法联网拉取镜像，需在旧服务器导出镜像（本目录未包含镜像包，需单独生成）：

```bash
# 在旧服务器执行
docker save infiniflow/ragflow:v0.17.2 mysql:8.0.39 elasticsearch:8.11.3 \
  quay.io/minio/minio:RELEASE.2023-12-20T01-00-02Z valkey/valkey:8 \
  -o ragflow_images.tar

# 复制到新服务器后加载
docker load -i ragflow_images.tar
```

加载镜像后，跳过「二.2 拉取镜像」步骤，直接执行后续步骤。
