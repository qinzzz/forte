# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for Selector. For selector functioning in a pipeline, see
`pipeline_test.py` instead.
"""
import unittest

from forte.data.multi_pack import MultiPack
from forte.data.selector import (
    NameMatchSelector,
    RegexNameMatchSelector,
    FirstPackSelector,
    AllPackSelector,
)


class SelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.multi_pack = MultiPack()

        data_pack1 = self.multi_pack.add_pack(ref_name="pack1")
        data_pack2 = self.multi_pack.add_pack(ref_name="pack2")
        data_pack3 = self.multi_pack.add_pack(ref_name="pack_three")

        data_pack1.pack_name = "1"
        data_pack2.pack_name = "2"
        data_pack3.pack_name = "Three"

    def test_name_match_selector(self) -> None:
        selector = NameMatchSelector()
        selector.initialize(
            configs={"select_name": "pack1"},
        )
        packs = selector.select(self.multi_pack)
        doc_ids = ["1"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)

    def test_name_match_selector_backward_compatability(self) -> None:
        selector = NameMatchSelector(select_name="pack1")
        packs = selector.select(self.multi_pack)
        doc_ids = ["1"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)

        selector = NameMatchSelector("pack1")
        packs = selector.select(self.multi_pack)
        doc_ids = ["1"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)

    def test_regex_name_match_selector(self) -> None:
        selector = RegexNameMatchSelector()
        selector.initialize(
            configs={"select_name": "^.*\\d$"},
        )
        packs = selector.select(self.multi_pack)
        doc_ids = ["1", "2"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)

    def test_regex_name_match_selector_backward_compatability(self) -> None:
        selector = RegexNameMatchSelector(select_name="^.*\\d$")
        packs = selector.select(self.multi_pack)
        doc_ids = ["1", "2"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)

        selector = RegexNameMatchSelector("^.*\\d$")
        packs = selector.select(self.multi_pack)
        doc_ids = ["1", "2"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)

    def test_first_pack_selector(self) -> None:
        selector = FirstPackSelector()
        packs = list(selector.select(self.multi_pack))
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0].pack_name, "1")

    def test_all_pack_selector(self) -> None:
        selector = AllPackSelector()
        packs = selector.select(self.multi_pack)
        doc_ids = ["1", "2", "Three"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.pack_name)


if __name__ == "__main__":
    unittest.main()
