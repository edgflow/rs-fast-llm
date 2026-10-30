//! Part 5: the inference server, built on channels. Users send their requests over one
//! channel, and every story comes back, piece by piece, over a channel of its own.

use crate::generate::generate_each;
use crate::live::Live;
use crate::math;
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// One request: a prompt, how many tokens to write, and where to send the story.
pub struct Ask {
    pub prompt: String,
    pub steps: usize,
    pub reply: Sender<String>,
}

/// The queue is full: try again later. (A web server would answer 503.)
#[derive(Debug, PartialEq)]
pub struct Busy;

pub struct Chat {
    asks: Sender<Ask>,
    workers: Vec<JoinHandle<usize>>,
}

impl Chat {
    /// Start `workers` threads that wait for requests and serve them, until shutdown.
    /// `queue`: how many requests may wait for a worker (None: no limit).
    pub fn start(
        w: Arc<Weights>,
        tok: Arc<Tokenizer>,
        workers: usize,
        queue: Option<usize>,
    ) -> Chat {
        let (asks, queue) = match queue {
            Some(room) => bounded::<Ask>(room),
            None => unbounded::<Ask>(),
        };
        let workers = (0..workers)
            .map(|_| {
                let (w, tok, queue) = (Arc::clone(&w), Arc::clone(&tok), queue.clone());
                thread::spawn(move || serve(&w, &tok, queue))
            })
            .collect();
        Chat { asks, workers }
    }

    /// Queue a request. Its story comes back on the receiver, one piece at a time.
    /// When the queue is full, `send` waits for room: backpressure on the user.
    pub fn ask(&self, prompt: &str, steps: usize) -> Receiver<String> {
        let (ask, pieces) = Ask::new(prompt, steps);
        self.asks.send(ask).expect("the workers are gone");
        pieces
    }

    /// The same, but when the queue is full, say so at once instead of waiting.
    pub fn try_ask(
        &self,
        prompt: &str,
        steps: usize,
    ) -> Result<Receiver<String>, Busy> {
        let (ask, pieces) = Ask::new(prompt, steps);
        match self.asks.try_send(ask) {
            Ok(()) => Ok(pieces),
            Err(TrySendError::Full(_)) => Err(Busy),
            Err(TrySendError::Disconnected(_)) => panic!("the workers are gone"),
        }
    }

    /// No more requests: the workers finish the queue and stop. How many each served.
    pub fn shutdown(self) -> Vec<usize> {
        drop(self.asks); // the last sender: every worker's loop ends once the queue is empty
        let workers = self.workers.into_iter();
        workers
            .map(|h| h.join().expect("a worker panicked"))
            .collect()
    }
}

impl Ask {
    /// A request, and the receiver its story will come back on.
    fn new(prompt: &str, steps: usize) -> (Ask, Receiver<String>) {
        let (reply, pieces) = unbounded();
        let prompt = prompt.to_string();
        (
            Ask {
                prompt,
                steps,
                reply,
            },
            pieces,
        )
    }
}

/// One worker: wait for a request, stream its story back, repeat.
fn serve(w: &Weights, tok: &Tokenizer, queue: Receiver<Ask>) -> usize {
    let mut served = 0;
    // waits while the queue is empty, and ends when it is empty and closed
    for ask in queue {
        let mut send = |piece: &str| ask.reply.send(piece.to_string()).is_ok();
        let live = Live::default();
        generate_each(
            w,
            tok,
            &ask.prompt,
            ask.steps,
            &math::matmul,
            &live,
            &mut send,
        );
        served += 1;
    } // each `ask.reply` is dropped at the end of its turn: that user's story is complete
    served
}
