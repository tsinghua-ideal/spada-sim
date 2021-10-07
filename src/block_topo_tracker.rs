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

    pub fn find_left(&self, cur_block: [usize; 2]) -> Option<(usize, [usize; 2])> {
        let row_pos = match self.row_s_list.binary_search(&cur_block[0]) {
            Ok(r) | Err(r) => r as i32 - 1,
        };
        if row_pos < 0 {
            return None;
        }
        let row_pos = row_pos as usize;

        let col_pos = match self.col_s_list[row_pos].binary_search(&cur_block[1]) {
            Ok(c) | Err(c) => c as i32 - 1,
        };

        if col_pos < 0 {
            return None;
        } else {
            return Some((
                self.token_list[row_pos][col_pos as usize],
                [cur_block[1], self.col_s_list[row_pos][col_pos as usize]],
            ));
        }
    }

    pub fn find_above(&self, cur_block: [usize; 2]) -> Option<(usize, [usize; 2])> {
        let row_pos = match self.row_s_list.binary_search(&cur_block[0]) {
            Ok(r) | Err(r) => r as i32 - 1,
        };

        if row_pos < 0 || self.col_s_list[row_pos as usize].len() == 0 {
            return None;
        }

        let row_pos = row_pos as usize;

        match self.col_s_list[row_pos].binary_search(&cur_block[1]) {
            Ok(c) => Some((
                self.token_list[row_pos][c],
                [self.row_s_list[row_pos], self.col_s_list[row_pos][c]],
            )),
            Err(c) => {
                let c_l = max(c - 1, 0);
                let c_r = min(c + 1, self.col_s_list[row_pos].len() - 1);
                if (cur_block[1] as i64 - self.col_s_list[row_pos][c_l] as i64).abs()
                    >= (self.col_s_list[row_pos][c_r] as i64 - cur_block[1] as i64).abs()
                {
                    return Some((
                        self.token_list[row_pos][c_r],
                        [self.row_s_list[row_pos], self.col_s_list[row_pos][c_r]],
                    ));
                } else {
                    return Some((
                        self.token_list[row_pos][c_l],
                        [self.row_s_list[row_pos], self.col_s_list[row_pos][c_l]],
                    ));
                }
            }
        }
    }

    pub fn add_block(&mut self, token: usize, anchor: [usize; 2]) {
        if anchor[1] == 0 {
            self.row_s_list.push(anchor[0]);
            self.col_s_list.push(vec![]);
            self.token_list.push(vec![]);
        }
        self.col_s_list.last_mut().unwrap().push(anchor[1]);
        self.token_list.last_mut().unwrap().push(token);
    }
}
