# Use a base image that guarantees the compiler is available
FROM gcc:latest

# 1. Create a dedicated low-privilege user
RUN useradd --create-home --shell /bin/bash sandboxuser

# 2. Set necessary permissions for the home directory
RUN chmod 755 /home/sandboxuser

# 3. Switch to the non-root user
USER sandboxuser

# 4. Set the working directory for code execution
WORKDIR /home/sandboxuser/code

# The default command will be overridden by the host
CMD ["/bin/bash"]