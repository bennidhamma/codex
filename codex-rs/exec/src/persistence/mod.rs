mod postgres;
mod redis;

use std::time::Duration;

pub use postgres::MessageRecord;
pub use postgres::PostgresStore;
pub use postgres::ToolInvocationEnd;
pub use postgres::ToolInvocationStart;
pub use redis::RedisPublisher;
pub use redis::RedisSubscriber;

pub async fn retry_with_backoff<F, Fut, T>(mut op: F, context: &str) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut delay = Duration::from_millis(200);
    let mut attempts = 0;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                attempts += 1;
                if attempts >= 6 {
                    return Err(err.context(format!("Exceeded retry attempts for {context}")));
                }
                tracing::warn!("Retrying {context} after error: {err:?} (attempt {attempts})");
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(10));
            }
        }
    }
}
