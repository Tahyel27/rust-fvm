use std::sync::{Arc, Condvar, Mutex};


struct Inner<T> {
    data: Option<T>,
    flag: bool,
    receiver_dropped: bool
}
pub struct Receiver<T> {
    inner: Arc<(Mutex<Inner<T>>, Condvar)>
}

pub struct Sender<T> {
    inner: Arc<(Mutex<Inner<T>>, Condvar)>
}

impl<T> Sender<T> {
    pub fn send(&self, data: T) -> Result<(), T> {
        let (lock, cvar) = &*self.inner;

        let mut inner = lock.lock().unwrap();

        if inner.receiver_dropped {
            return Err(data);
        }

        inner.data = Some(data);
        inner.flag = true;

        cvar.notify_one();
        Ok(())
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> Option<T> {
        
        let (lock, cvar) = &*self.inner;

        let mut inner = lock.lock().unwrap();

        while !inner.flag {
            inner = cvar.wait(inner).unwrap();
        }
        
        inner.flag = false;
        inner.data.take()
    }

    pub fn try_recv(&self) -> Option<T> {
        let (lock, cvar) = &*self.inner;
        let mut inner = lock.lock().unwrap();

        inner.data.take()
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.inner;
        let mut inner = lock.lock().unwrap();

        inner.receiver_dropped = true;
    }
}

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let inner = Inner{ data: None, flag: false, receiver_dropped: false };

    let arc = Arc::new((Mutex::new(inner), Condvar::new()));
    let arc2 = Arc::clone(&arc);

    (Sender{inner: arc}, Receiver{inner: arc2})
}