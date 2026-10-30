//! Two locks built from atomics: a spinlock and a ticket lock. Eight threads take each
//! one as often as they can for one second. Does every thread get its turn?

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::thread;
use std::time::Duration;

const THREADS: usize = 8;

/// A spinlock: one flag, taken with a compare-and-swap. Waiting threads stay awake and
/// keep trying, instead of going to sleep like a Mutex.
pub struct SpinLock<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: only the thread that holds the lock reaches the value
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub fn new(value: T) -> Self {
        let locked = AtomicBool::new(false);
        let value = UnsafeCell::new(value);
        SpinLock { locked, value }
    }

    pub fn lock(&self) -> SpinGuard<'_, T> {
        // false → true: only one thread can win; Acquire, to see the last holder's writes
        while self
            .locked
            .compare_exchange_weak(false, true, Acquire, Relaxed)
            .is_err()
        {
            spin_loop();
        }
        SpinGuard { lock: self }
    }
}

pub struct SpinGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> Deref for SpinGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.lock.value.get() } // we hold the lock
    }
}

impl<T> DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.value.get() } // we hold the lock
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Release); // our writes go out with the unlock
    }
}

/// A ticket lock, like the queue at a deli counter: take a number, wait until it is
/// served. First come, first served.
pub struct TicketLock<T> {
    next: AtomicUsize,
    serving: AtomicUsize,
    value: UnsafeCell<T>,
}

// SAFETY: only the thread whose ticket is being served reaches the value
unsafe impl<T: Send> Sync for TicketLock<T> {}

impl<T> TicketLock<T> {
    pub fn new(value: T) -> Self {
        let (next, serving) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let value = UnsafeCell::new(value);
        TicketLock {
            next,
            serving,
            value,
        }
    }

    pub fn lock(&self) -> TicketGuard<'_, T> {
        let ticket = self.next.fetch_add(1, Relaxed); // every thread gets its own number
        while self.serving.load(Acquire) != ticket {
            spin_loop();
        }
        TicketGuard { lock: self }
    }
}

pub struct TicketGuard<'a, T> {
    lock: &'a TicketLock<T>,
}

impl<T> Deref for TicketGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.lock.value.get() } // our ticket is being served
    }
}

impl<T> DerefMut for TicketGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.value.get() } // our ticket is being served
    }
}

impl<T> Drop for TicketGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.serving.fetch_add(1, Release); // the next number, please
    }
}

/// Every thread calls `take(me)` in a loop for one second.
fn contest(take: impl Fn(usize) + Sync) {
    let stop = AtomicBool::new(false);
    thread::scope(|s| {
        for me in 0..THREADS {
            let (take, stop) = (&take, &stop);
            s.spawn(move || {
                while !stop.load(Relaxed) {
                    take(me);
                }
            });
        }
        thread::sleep(Duration::from_secs(1));
        stop.store(true, Relaxed);
    });
}

fn report(name: &str, turns: [usize; THREADS]) {
    let total: usize = turns.iter().sum();
    let (most, least) = (turns.iter().max().unwrap(), turns.iter().min().unwrap());
    println!("{name}: {total} turns in 1 s, per thread {turns:?}");
    println!(
        "{name}: the busiest thread got {:.1}x the least",
        *most as f64 / *least.max(&1) as f64
    );
}

fn main() {
    let spin = SpinLock::new([0usize; THREADS]);
    contest(|me| spin.lock()[me] += 1);
    report("spinlock", *spin.lock());

    let ticket = TicketLock::new([0usize; THREADS]);
    contest(|me| ticket.lock()[me] += 1);
    report("ticket lock", *ticket.lock());
}
