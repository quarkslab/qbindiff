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

"""Collection of refinement passes

This module contains the interface to implement a pass to refine the similarity
matrix.
There are already several standard implementations you can choose from, most
notably the FeaturePass that uses the result of the features to populate the
content of the similarity matrix.
"""

from qbindiff.passes.base import FeaturePass
from qbindiff.passes.utils import ZeroPass
