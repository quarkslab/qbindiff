# -*- encoding: utf8 -*-
#
# QBinViz plugin to vizualize qbindiff diffing
#
# Copyright (c) 2018 Quarkslab
# Author: Robin David <rdavid@quarkslab.com>
import os
import os.path
import json
import threading

import ida_graph
import ida_idaapi
import ida_kernwin
import ida_lines
import ida_moves
import ida_auto
import ida_funcs

from qbindiff.loader.program import Program
from qbindiff.loader.function import Function
from qbindiff.loader.types import LoaderType
from qbindiff.differ.matching import Matching

VIEW_A = "IDA View-A"
VIEW_B = "QBinViz View-B"


class Hooker(ida_kernwin.View_Hooks):
    def __init__(self):
        ida_kernwin.View_Hooks.__init__(self)
        self.mapping = None
        self.program = None
        self.mirror_widget = None
        self.curr_f = 0
        self.switch_button = False

    def is_initialized(self):
        return self.mapping is not None

    def reset(self):
        self.mapping = None
        self.program = None
        self.unhook()

    def set_mapping(self, mapping):
        self.mapping = mapping

    def set_program(self, program):
        self.program = program

    def view_close(self, view):
        name = ida_kernwin.get_widget_title(view)
        if name in [VIEW_A, VIEW_B]:
            print("Closing: %s (thus unhook)" % name)
            self.unhook()
            # TODO: Cleaning program and mapping ?

    def update_widget_b(self, view_a):
        # Make sure we are in the same function
        place_a, _, _ = ida_kernwin.get_custom_viewer_place(view_a, False)
        view_b = ida_kernwin.find_widget(VIEW_B)
        if view_b is None:
            print(VIEW_B + " is None stop")
            return
        ida_kernwin.jumpto(view_b, place_a, -1, -1)

        # and that we show the right place (slightly zoomed out)
        widget_a_center_gli = ida_moves.graph_location_info_t()
        if ida_graph.viewer_get_gli(widget_a_center_gli, view_a, ida_graph.GLICTL_CENTER):
            widget_b_center_gli = ida_moves.graph_location_info_t()
            widget_b_center_gli.orgx = widget_a_center_gli.orgx
            widget_b_center_gli.orgy = widget_a_center_gli.orgy
            widget_b_center_gli.zoom = widget_a_center_gli.zoom  # * 0.5
            ida_graph.viewer_set_gli(view_b, widget_b_center_gli, ida_graph.GLICTL_CENTER)

    def view_loc_changed(self, view, now, was):
        name = ida_kernwin.get_widget_title(view)
        if name == VIEW_A:
            if self.switch_button:
                self.switch_button = False
                return
            now_ea = now.place().toea()
            now_ea_fun = ida_funcs.get_func(now_ea).start_ea
            was_ea = was.place().toea()
            was_ea_fun = ida_funcs.get_func(was_ea).start_ea
            if was_ea_fun != now_ea_fun:
                print("Function changed (0x%x -> 0x%x)" % (self.curr_f, now_ea_fun))
                if now_ea_fun in self.mapping:
                    target_fun_ea = self.mapping[now_ea_fun]
                    self.mirror_widget.switch_to_function(self.program[target_fun_ea])
                else:
                    print("Function 0x%x does not have sibling." % now_ea_fun)
                    self.mirror_widget.switch_to_unknown()
                ida_graph.viewer_fit_window(view)  # Can't as it activate max recursion
                self.switch_button = True
                ida_graph.viewer_fit_window(ida_kernwin.find_widget(VIEW_B))  # Fit VIEW_B
            else:
                pass  # Just update the position of the view B
                # print("Update widget B")
                # self.update_widget_b(view)

    def view_switched(self, view, rt):
        print("View switched")
        pass  # TODO: do something when switching to listing view ?

    # TODO: Write other handle even if not used


class VirtualFunctionViewer(ida_graph.GraphViewer):
    def __init__(self, title, close_open=False):
        ida_graph.GraphViewer.__init__(self, title, close_open=close_open)

    def switch_to_function(self, f):
        m, n = zip(*(((f_addr, i), bb) for i, (f_addr, bb) in enumerate(f.items())))
        mapping = dict(m)
        self.Clear()
        self._nodes = ["\n".join(str(x) for x in y) for y in n]
        self._edges = map(lambda x: (mapping[x[0]], mapping[x[1]]), f.edges)
        # RMQ: if too slow precomputing these lists!
        self.Refresh()
        # ida_graph.viewer_fit_window(ida_kernwin.find_widget(self._title))

    def switch_to_unknown(self):
        self.Clear()
        self._nodes = ["NO MATCH"]
        self._edges = []
        self.Refresh()

    def OnRefresh(self):
        print("On refresh called")
        return True

    def OnGetText(self, node_id):
        return self._nodes[node_id]

    def Show(self):
        if not ida_graph.GraphViewer.Show(self):
            return False
        return True

    def OnClose(self):
        print("qBinViz closed!")

    def OnSelect(self, node_id):
        return True

    def OnClick(self, node_id):
        return True


class FileSelectionForm(ida_kernwin.Form):
    def __init__(self):
        self.primary = ida_kernwin.Form.FileInput(open=True, swidth=25)
        self.secondary = ida_kernwin.Form.FileInput(open=True, swidth=25)
        self.primaryD = ida_kernwin.Form.DirInput(swidth=25)
        self.secondaryD = ida_kernwin.Form.DirInput(swidth=25)
        self.loader_items = sorted(x.name for x in LoaderType)

        ida_kernwin.Form.__init__(
            self,
            r"""STARTITEM 0
BUTTON YES* Ok
BUTTON CANCEL Cancel
QBinViz

{FormChangeCb}
QBinDiff vizualization configuration:
<Loader : {iLoader}>
<Primary : {iPrimary}>
<Secondary : {iSecondary}>
<Matching : {iMatching}>
<Primary is current binary:{rIsCurrent}>{cGroupSame}>
""",
            {
                "iLoader": ida_kernwin.Form.DropdownListControl(
                    items=sorted(x.name for x in LoaderType), readonly=True, selval=0
                ),
                "iPrimary": self.primary,
                "iSecondary": self.secondary,
                "iMatching": ida_kernwin.Form.FileInput(open=True, swidth=25),
                #'iDir': ida_kernwin.Form.DirInput(swidth=23.2),
                "cGroupSame": ida_kernwin.Form.ChkGroupControl(("rIsCurrent",)),
                #'cBox': ida_kernwin.Form.ChkGroupControl(('rSameFile')),
                "FormChangeCb": ida_kernwin.Form.FormChangeCb(self.OnFormChange),
            },
        )
        self.Compile()

        self.rIsCurrent.checked = True

    def __to_loadertype(self, v):
        return LoaderType[self.loader_items[v]]

    def OnFormChange(self, fid):
        if fid == -1:  # The form is being initialized
            self.EnableField(self.iPrimary, False)
        elif fid == -2:  # Ok has been clicked
            print("Data received")
        elif fid == self.cGroupSame.id:
            val = self.GetControlValue(self.cGroupSame)
            print("Checkbox changed:", val)
            self.EnableField(self.iPrimary, False if val else True)
        elif fid == self.iLoader.id:
            v = self.GetControlValue(self.iLoader)
            typ = LoaderType[self.loader_items[v]]
            if typ in [LoaderType.binexport, LoaderType.diaphora]:
                print("binexport/diaphora set: put file mode")
                self.controls["iPrimary"] = self.primary
                self.controls["iSecondary"] = self.secondary
            else:
                assert False
            self.iLoader.value = v
            self.Compile()
            self.Close(1)
            self.Execute()
        else:
            print("fid: %d" % fid)
        return 1

    def get_primary_filepath(self):
        return self.iPrimary.value

    def get_secondary_filepath(self):
        return self.iSecondary.value

    def get_matching_filepath(self):
        return self.iMatching.value

    def is_primary_local(self):
        return bool(self.cGroupSame.value)

    def get_type(self):
        return LoaderType[self.loader_items[self.iLoader.value]]


class QBinVizPlugin(ida_idaapi.plugin_t):
    wanted_name = "qBinViz"
    wanted_hotkey = "Shift-d"
    flags = 0
    comment = ""
    help = "qBinViz: Vizualizing qBinDiff diffs in IDA Pro"

    def init(self):
        print("Initialize qBinViz")
        self.hooker = Hooker()
        self.matching = None
        return ida_idaapi.PLUGIN_KEEP

    def run(self, args):
        if self.hooker.is_initialized():
            print("qBinViz reset configuration")
            self.hooker.reset()
        else:
            print("Start qBinViz")

        form = FileSelectionForm()
        res = form.Execute()
        if res:
            print("Load data .. (please wait)")
            if form.is_primary_local():
                pass  # primary = Program(LoaderType.ida.name)  # Not used when using the local IDA
            else:
                print("Non-local primary are not yet supported")
                return
                # primary = Program(form.get_type().name, form.get_primary_filepath())
            f_path = form.get_secondary_filepath()
            typ = form.get_type()
            if typ == LoaderType.binexport:
                secondary = Program(typ.name, f_path)
            else:
                print("Type not supported yet")
                return
            self.matching = Matching(file=form.get_matching_filepath())
            self.hooker.set_program(secondary)
            self.hooker.set_mapping(self.matching.matching)
            self.init_views()
        else:
            print("qBinViz cancelled")
        form.Free()

    def init_views(self):
        # Put both views in graph mode
        # TODO: If not local primary open a VirtualFunctionViewer (like VIEW_B)
        widget_a = ida_kernwin.find_widget(VIEW_A)
        if not widget_a:
            widget_a = ida_kernwin.open_disasm_window("A")
        ida_kernwin.set_view_renderer_type(widget_a, ida_kernwin.TCCRT_GRAPH)
        # ida_graph.viewer_fit_window(self.widget_a)

        widget_b = ida_kernwin.find_widget(VIEW_B)
        if not widget_b:
            print("Create VirtualFunctionViewer for %s" % VIEW_B)
            self.hooker.mirror_widget = VirtualFunctionViewer(VIEW_B)
            self.hooker.mirror_widget.Show()
            widget_b = ida_kernwin.find_widget(VIEW_B)
            # self.widget_b = ida_kernwin.open_disasm_window('B')
        self.hooker.view_activated(widget_a)  # to update widget b automatically

        # self.widget_b = ida_kernwin.find_widget(self.VIEW_B)  # TODO: To remove when it will be working
        ida_kernwin.set_view_renderer_type(widget_b, ida_kernwin.TCCRT_GRAPH)

        # Set view B next to view A
        ida_kernwin.set_dock_pos(VIEW_B, VIEW_A, ida_kernwin.DP_RIGHT)

        # Start hooking and update view_a to get nice view
        self.hooker.hook()
        ida_graph.viewer_fit_window(widget_a)
        ida_graph.viewer_fit_window(widget_b)

    def term(self):
        print("Terminate qBinViz")
        self.hooker.unhook()


qBinViz = None


def PLUGIN_ENTRY():
    global qBinViz
    qBinViz = QBinVizPlugin()
    return qBinViz


# If the script is launched through Ctrl+F7
if __name__ == "__main__":
    qBinViz = QBinVizPlugin()
    qBinViz.init()
    qBinViz.run([])
