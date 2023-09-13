# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Features class manager
"""


class FeatureKeyManagerClass:
    """
    Singleton class that assigns a unique number to each main_key/sub_key.
    (Internal usage only).
    """

    def __init__(self):
        self.subkeys = {}
        self.mainkeys = {}

    def add(self, main_key: str, sub_key: str | None = None) -> None:
        """
        Add the main_key and optionally the sub_key to the key manager.
        """

        self.mainkeys[main_key] = len(self.mainkeys)
        self.subkeys.setdefault(main_key, {})
        if sub_key and sub_key not in self.subkeys[main_key]:
            self.subkeys[main_key][sub_key] = len(self.subkeys[main_key])

    def get(self, main_key: str, sub_key: str | None = None) -> int:
        """
        Get the unique number given the main_key or the sub_key.
        """

        if sub_key:
            return self.subkeys[main_key][sub_key]
        else:
            return self.mainkeys[main_key]

    def get_cumulative_size(self, main_key_list: list[str]) -> int:
        """
        Returns the cumulative size of all the main_keys
        """

        return sum(
            size if (size := len(self.subkeys[main_key])) > 0 else 1 for main_key in main_key_list
        )

    def get_size(self, main_key: str) -> int:
        """
        Returns the size of the main_key specified.
        """

        if (size := len(self.subkeys[main_key])) > 0:
            return size
        else:
            return 1


# Singleton object
FeatureKeyManager = FeatureKeyManagerClass()

# Make it a little bit harder to use this class the wrong way
del FeatureKeyManagerClass
