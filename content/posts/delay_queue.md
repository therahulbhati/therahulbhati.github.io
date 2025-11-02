---
date: '2025-10-21T13:47:46+05:30'
draft: false
title: 'Delay Queues: Do It Later, Reliably'
tags: 
  - 'Delay Queue'
  - 'Job Queue'
  - 'Background Jobs'
  - 'Asynchronous Processing'
  - 'Distributed Systems'
  - 'System Design'
  - 'Redis'
  - 'Postgres'
  - 'AWS SQS'
  - 'Message Queue'
categories: 
  - 'Backend Engineering'
  - 'Distributed Systems'
cover:
    image: /images/delay_queue_1.png
    alt: 'Illustration of a delay queue dashboard showing scheduled jobs with different statuses and workers processing them on a timeline'
    caption: 'Choosing the right delay queue architecture for your use case'
---

## What is a delay queue?
A delay queue holds a message/job until a specified delay time in the future, then makes it available for processing. It's the simplest way to schedule "do this later" without blocking workers or rolling your own cron logic.

## Why use a delay queue?
- **Retries with backoff** — retry failed tasks after a delay.
- **Scheduled tasks** — run at specific times (e.g., reminder emails).
- **Rate smoothing** — spread out work to avoid thundering herds.

## When to choose what

### 1. Broker-based Delay Queues
Built-in support for delayed messages in brokers.

**Cloud-managed**
- **AWS SQS** (`DelaySeconds`, per-message up to 15 min)
- **Azure Service Bus** (scheduled messages)
- **Google Cloud Tasks** (scheduled delivery)

**Open-source**
- **Apache Pulsar** (delayed messages)
- **RabbitMQ** (delayed-message plugin)

**Best for**:
- Static/simple delays  
- Operational simplicity over flexibility  
- Delay ranges: seconds → hours *(SQS: 15 min/message; Service Bus/Pulsar: much longer)*

**Tradeoffs**:
- Limited introspection: hard to list/search/cancel/reschedule after enqueue  
- Cloud-managed = zero ops but vendor lock-in  
- Open-source = portable but you run the infra 

### 2. Redis ZSET Scheduler
Use a Redis **sorted set** with score = execution timestamp. Fast lookups; easy cancel/reschedule.

**Best for**:
- High-throughput scheduling (10K+ jobs/sec)  
- Cancellable delays & "list upcoming jobs" APIs  
- Dynamic rescheduling (snooze/postpone)  
- Sub-second granularity 

**Implementation**:
```bash
# Schedule a job
ZADD delays <execution_timestamp> <job_id>

# Fetch due jobs (batch of 100)
ZRANGEBYSCORE delays -inf <now> LIMIT 0 100

# Remove once picked up
ZREM delays <job_id>
```
**Tradeoffs**:
- Single point of failure unless Redis is clustered or replicated.
- In-memory volatility: Data can be lost on crash if AOF/RDB not configured properly.
- External Execution Required: Redis only tracks when to run jobs — execution must be handled by polling workers.


### 3. Postgres + SKIP LOCKED
Store jobs in your main database. Poll with `FOR UPDATE SKIP LOCKED`.

**Best for**:
- Atomic job scheduling + business logic ("cancel order and schedule refund" in one DB transaction)
- Teams already using Postgres — no new infrastructure.
- Strong consistency and strict ordering requirements.
- Moderate throughput workloads (<10K jobs/sec).

**Implementation**:
```sql
SELECT * FROM jobs
WHERE run_at <= NOW() AND status = 'pending'
ORDER BY run_at
LIMIT 100
FOR UPDATE SKIP LOCKED;
```

**Why `SKIP LOCKED`?**
Multiple workers poll concurrently without lock contention. Each grabs a batch, skips locked rows.

**Tradeoffs**:
- Polling interval creates latency floor (5s poll = up to 5s jitter). *Use `LISTEN/NOTIFY` to wake workers immediately on job insert.* ([credit: @alexpovel](https://github.com/alexpovel))
- Index pressure on high churn.
- Not for sub-second precision.

### 4. DynamoDB TTL
AWS-native approach using DynamoDB's Time To Live (TTL) feature combined with DynamoDB Streams.

**Best for**:
- Serverless architectures on AWS.
- Long delays (hours to days).
- Variable/unpredictable load patterns

**Implementation**:
```javascript
// Schedule item with TTL
await dynamodb.putItem({
  TableName: 'delayed_jobs',
  Item: {
    job_id: 'order-123',
    ttl: Math.floor(Date.now() / 1000) + delay_seconds,
    payload: { /* job data */ }
  }
});

// Lambda triggered by DynamoDB Streams on deletion
```
**Tradeoffs**:
- TTL deletion is eventually consistent (typically within 48 hours, but usually minutes)
- Not suitable for precise timing needs.
- AWS lock-in.

### 5. Timer Wheels / Bucketed Timers
An in-memory data structure optimized for millions of delays per second.

**Examples**:
- Netty’s HashedWheelTimer
- Kafka broker request purgatory (hierarchical timing wheels) [reference](https://www.confluent.io/blog/kafka-internals-request-purgatory/)

**Best for**:
- Session timeouts (30-sec idle = disconnect)
- Rate limiter token bucket refills
- Game server tick-based events
- Short delays (<1 hour), acceptable precision ~100ms

**Why it's fast?**
- O(1) insertions using a circular wheel of buckets
- Batched execution of timers in the same bucket
- Trades precision for throughput.

**Tradeoffs**:
- In-memory only(not durable)
- Perfect for ephemeral delays where loss is acceptable.

### 6. Workflow Engines
Full orchestration engines with built-in delay scheduling.

**Examples**:
- Temporal
- Camunda

**Best for**:
- Multi-step processes with delays between steps (e.g., Abandoned cart: 1h idle → +2h reminder → +24h reminder (+incentive) → close.).
- Human approval steps with timeouts
- Microservice orchestration with retries and delays
- Audit trails and visual workflow editors

**Tradeoffs**:
- Heavy infrastructure and operational complexity.
- Overkill for simple delay needs, but perfect when delays are part of larger state workflows.
- Steep learning curve
- Resource-intensive (separate database, workers, UI)

## Delivery Semantics & Idempotency
Most delay queue implementations provide **at-least-once** delivery guarantees. This means jobs may be delivered multiple times due to:
- Worker crashes before acknowledging
- Network timeouts during acknowledgment
- Duplicate scheduling

**Design for Idempotency**
Your job handlers should be safe to retry. Use unique job IDs, database constraints, or distributed locks to ensure processing a job multiple times has no adverse effects.

**Exactly-once Delivery**
Kafka (with transactions) and Pulsar (with deduplication) can provide stronger guarantees, but at the cost of complexity.

**Dead Letter Queues (DLQs)**
- Most delay queue systems support DLQs to isolate jobs that permanently fail after exhausting retries
- When a job fails repeatedly (e.g., 3-5 attempts with exponential backoff), it's moved to the DLQ for manual inspection
- Prevents poisonous messages from blocking healthy jobs
- DLQ growth indicates systematic issues like bad code deploys, external service outages, or malformed data

## Observability
Regardless of the delay queue you choose, implement monitoring and alerting for:
- **Queue depth**: Number of pending jobs (alert on unexpected growth)
- **Lag**: Time between scheduled and actual execution
- **Throughput**: Jobs processed per second
- **Failure rates**: Percentage of jobs that fail and require retries
- **Age of oldest job**: Detect stuck/stale jobs
- **Dead letter queue (DLQ) depth**: Jobs that exceeded retry limits

## Quick Decision Matrix

| Your constraint                        | Go with                    |
| -------------------------------------- | -------------------------- |
| Fully managed (AWS)                    | **SQS**                    |
| Fully managed (GCP)                    | **Cloud Tasks**            |
| Fully managed (Azure)                  | **Service Bus**            |
| Fully managed, avoid vendor lock-in    | **Managed Pulsar**         |
| Need speed + easy cancellation/inspect | **Redis ZSET**             |
| Need transactional consistency         | **Postgres + SKIP LOCKED** |
| Serverless AWS + long delays           | **DynamoDB TTL**           |
| Short-lived delays, loss acceptable    | **Timer wheels**           |
| 100K+ delays/sec, ephemeral OK         | **Timer wheels**           |
| Multi-step workflows with state        | **Temporal / Camunda**     |

## Final Thoughts
Choose based on your requirements: cloud-managed for simplicity, Redis/Postgres for speed and control, workflow engines for complex orchestration. The best delay queue fits your existing stack and operational maturity.

Don't build your own delay queue. Use an existing solution unless you have very specific requirements that nothing else can satisfy.
