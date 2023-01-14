use crate::trace_println;
use std::cmp::{max, min};

pub struct BlockTopoTracker {
    pub row_s_list: Vec<usize>,
    pub col_s_list: Vec<Vec<usize>>,
    pub token_list: Vec<Vec<usize>>,
}

impl BlockTopoTracker {
    pub fn new() -> BlockTopoTracker {
        BlockTopoTracker {
            row_s_list: vec![],
            col_s_list: vec![],
            token_list: vec![],
        }
    }

    pub fn add_block(&mut self, token: usize, anchor: [usize; 2]) {
        match self.row_s_list.binary_search(&anchor[0]) {
            Ok(r) => {
                let c = self.col_s_list[r]
                    .binary_search(&anchor[1])
                    .unwrap_or_else(|x| x);
                self.col_s_list[r].insert(c, anchor[1]);
                self.token_list[r].insert(c, token);
            }
            Err(r) => {
                self.row_s_list.insert(r, anchor[0]);
                self.col_s_list.insert(r, vec![anchor[1]]);
                self.token_list.insert(r, vec![token]);
            }
        }
    }

    pub fn find_left(&self, cur_block: [usize; 2]) -> Option<(usize, [usize; 2])> {
        // Find the most near left in recorded blocks.
        if self.row_s_list.len() == 0 {
            return None;
        }
        let mut cur_row_pos = self.row_s_list.binary_search(&cur_block[0]).map_or_else(
            |x| min(max(self.row_s_list.len() as i32 - 1, 0), x as i32),
            |x| x as i32,
        );
        while cur_row_pos >= 0 {
            trace_println!("cur_block: {:?} cur_row_pos: {}", cur_block, cur_row_pos);
            let row_pos = cur_row_pos as usize;
            let col_pos = match self.col_s_list[row_pos].binary_search(&cur_block[1]) {
                Ok(c) | Err(c) => c as i32 - 1,
            };

            if col_pos < 0 {
                cur_row_pos -= 1;
            } else {
                let col_pos = col_pos as usize;
                let row_idx = self.row_s_list[row_pos];
                let col_idx = self.col_s_list[row_pos][col_pos];
                let token = self.token_list[row_pos][col_pos];
                return Some((token, [row_idx, col_idx]));
            }
        }

        return None;
    }

    pub fn find_above(&self, cur_block: [usize; 2]) -> Option<(usize, [usize; 2])> {
        if self.row_s_list.len() == 0 {
            return None;
        }
        let cur_row_pos = self.row_s_list.binary_search(&cur_block[0]).map_or_else(
            |x| x as i32,
            |x| min(max(self.row_s_list.len() as i32 - 1, 0), x as i32),
        );
        if cur_row_pos == 0 {
            return None;
        }

        let row_pos = (cur_row_pos - 1) as usize;
        let col_pos = match self.col_s_list[row_pos].binary_search(&cur_block[1]) {
            Ok(c) => c,
            Err(c) => {
                if c == self.col_s_list[row_pos].len() {
                    c - 1
                } else {
                    if self.col_s_list[row_pos][c] - cur_block[1]
                        < cur_block[1] - self.col_s_list[row_pos][c - 1]
                    {
                        c
                    } else {
                        c - 1
                    }
                }
            }
        };
        let row_idx = self.row_s_list[row_pos];
        let col_idx = self.col_s_list[row_pos][col_pos];
        let token = self.token_list[row_pos][col_pos];

        return Some((token, [row_idx, col_idx]));
    }
}
