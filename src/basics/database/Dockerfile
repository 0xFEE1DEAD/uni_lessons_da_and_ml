FROM postgres:17

# Variables used by postgres image - defining user, password and database name
ENV POSTGRES_DB="postgres"
ENV POSTGRES_USER="postgres"
ENV POSTGRES_PASSWORD="postgres"

# We need first download archive with DVD Rental backup and then extract it
ENV ARCHIVE_NAME="dvdrental.zip"
ENV BACKUP_NAME="dvdrental.tar"
ENV URL="https://neon.tech/postgresqltutorial/${ARCHIVE_NAME}"

RUN apt-get update && apt-get install -y wget unzip

RUN wget -nv "${URL}" -O "/tmp/${ARCHIVE_NAME}" \
  && unzip -q "/tmp/${ARCHIVE_NAME}" -d /tmp

COPY restoredb.sh /docker-entrypoint-initdb.d/