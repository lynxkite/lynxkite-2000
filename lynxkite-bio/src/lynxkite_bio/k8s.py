"""Tools for starting and stopping Docker containers on Kubernetes.

A test setup for this feature:

```bash
# Start minikube with GPU support:
minikube start --driver docker --container-runtime docker --gpus all
# Make the services accessible:
minikube tunnel
```

Use `k8s.needs()` to declare a Kubernetes dependency for an operation. For example:

```python
@op("Ask LLM", slow=True)
@k8s.needs(
    name="vllm-for-ask-llm-op",
    image="vllm/vllm-openai:latest",
    port=8000,
    args=["--model", "google/gemma-3-1b-it"],
    health_probe="/health",
    forward_env=["HUGGING_FACE_HUB_TOKEN"],
    storage_path="/root/.cache/huggingface",
    storage_size="10Gi",
)
def ask_llm(df: pd.DataFrame, *, question: ops.LongStr):
    ip = k8s.get_ip("vllm-for-ask-llm-op")
    client = openai.OpenAI(api_key="EMPTY", base_url=f"http://{ip}/v1")
    # ...
```
"""

import functools
import os
import queue
import threading
import time
import httpx
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def _run(
    *,
    name,
    image,
    port,
    namespace,
    storage_size,
    storage_path,
    health_probe,
    forward_env,
    **kwargs,
):
    print(f"Starting {name} in namespace {namespace}...")
    volume_mounts = []
    volumes = []
    if storage_size:
        pvc_name = f"{name}-data-volume"
        if not _pvc_exists(pvc_name, namespace):
            _create_pvc(pvc_name, size=storage_size, namespace=namespace)
        volume_mounts.append(
            client.V1VolumeMount(
                name=pvc_name,
                mount_path=storage_path,
            )
        )
        volumes.append(
            client.V1Volume(
                name=pvc_name,
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc_name,
                ),
            )
        )
    # Forward local environment variables to the container.
    kwargs.setdefault("env", []).extend(
        [{"name": name, "value": os.environ[name]} for name in forward_env]
    )
    container = client.V1Container(
        name=name,
        image=image,
        ports=[client.V1ContainerPort(container_port=port)],
        volume_mounts=volume_mounts,
        **kwargs,
    )
    if health_probe:
        container.readiness_probe = client.V1Probe(
            http_get=client.V1HTTPGetAction(path=health_probe, port=port, scheme="HTTP"),
        )
    deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels={"app": name}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": name}),
                spec=client.V1PodSpec(
                    volumes=volumes,
                    containers=[container],
                ),
            ),
        ),
    )
    apps_v1 = client.AppsV1Api()
    apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)

    service_name = f"{name}-service"
    service = client.V1Service(
        metadata=client.V1ObjectMeta(name=service_name, labels={"app": name}),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(protocol="TCP", port=80, target_port=port)],
            type="LoadBalancer",
        ),
    )
    core_v1 = client.CoreV1Api()
    core_v1.create_namespaced_service(namespace=namespace, body=service)


def _stop(name, namespace="default"):
    print(f"Stopping {name} in namespace {namespace}...")
    apps_v1 = client.AppsV1Api()
    apps_v1.delete_namespaced_deployment(name, namespace)
    service_name = f"{name}-service"
    core_v1 = client.CoreV1Api()
    core_v1.delete_namespaced_service(service_name, namespace)


def get_ip(name: str, namespace: str = "default", timeout: int = 3600, interval: int = 1) -> str:
    """Look up the IP address where the operation can access the service."""
    service_name = f"{name}-service"
    core_v1 = client.CoreV1Api()
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            svc = core_v1.read_namespaced_service(service_name, namespace)
            ingress = svc.status.load_balancer.ingress
            if ingress:
                ip = ingress[0].ip or ingress[0].hostname
                if ip:
                    if _can_connect(ip):
                        return ip
        except ApiException as e:
            if e.status != 404:
                raise
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for external IP of service '{service_name}'")


def _can_connect(ip: str) -> bool:
    try:
        httpx.get(f"http://{ip}/")
        return True
    except httpx.RequestError:
        return False


def _is_running(name: str, namespace: str = "default") -> bool:
    apps_v1 = client.AppsV1Api()
    try:
        apps_v1.read_namespaced_deployment(name, namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        else:
            raise


def _stop_if_running(name, namespace="default"):
    if _is_running(name, namespace):
        _stop(name, namespace)


def _create_pvc(name, size="1Gi", namespace="default"):
    core_v1 = client.CoreV1Api()
    pvc = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(requests={"storage": size}),
        ),
    )
    core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)


def _pvc_exists(name: str, namespace: str = "default") -> bool:
    core_v1 = client.CoreV1Api()
    try:
        core_v1.read_namespaced_persistent_volume_claim(name=name, namespace=namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        else:
            raise


def needs(
    name: str,
    image: str,
    port: int,
    args: list = None,
    env: list = None,
    forward_env: list = None,
    health_probe: str = None,
    storage_size: str = None,
    storage_path: str = "/data",
    namespace: str = "default",
):
    """Use this decorator to configure a microservice that the operation depends on.
    LynxKite will manage the lifecycle of the microservice for you.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            config.load_kube_config()
            _using(
                name=name,
                image=image,
                port=port,
                args=args or [],
                env=env or [],
                forward_env=forward_env or [],
                health_probe=health_probe,
                storage_size=storage_size,
                storage_path=storage_path,
                namespace=namespace,
            )
            try:
                return func(*func_args, **func_kwargs)
            finally:
                _stop_using(name, namespace)

        return wrapper

    return decorator


_USER_COUNTERS = {}


def _using(name, **kwargs):
    q = _USER_COUNTERS.setdefault(name, queue.Queue(-1))
    q.put(1)
    try:
        if not _is_running(name):
            _run(name=name, **kwargs)
    except Exception as e:
        q.get()
        raise e


def _stop_using(name, namespace):
    q = _USER_COUNTERS[name]
    q.get()
    if q.empty():
        _stop_later(name, namespace)


def _stop_later(name, namespace):
    q = _USER_COUNTERS[name]

    def stop():
        time.sleep(6000)
        if q.empty():
            # Nobody started the service in the meantime.
            _stop(name, namespace)

    t = threading.Thread(target=stop)
    t.start()
