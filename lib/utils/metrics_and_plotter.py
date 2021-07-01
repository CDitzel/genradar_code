import visdom
import numpy as np
import torch


class VisdomPlotter:
    """Stats to Visdom"""

    def __init__(self, ip_ts_desc="", port=8097, env_name="main"):
        self.ip = "134.60.29.93"
        # self.env = env_name
        self.env = ip_ts_desc
        try:
            print("Trying to establish visdom connection...")
            self.viz = visdom.Visdom(server=self.ip, port=8097, env=self.env)
            if not self.viz.check_connection(timeout_seconds=3):
                raise Exception("No connection could be formed quickly")
        except Exception as e:
            print(f"Couldnt connect to visdom")
        else:
            print("Succesfully set up VisdomPlotter")
            self.plots = {}
            # self.images = {}
            self.ip_ts_desc = ip_ts_desc
        finally:
            print(f"Navigate to {self.ip}:{port} in your browser")
            # updatetextwindow = self.viz.text("Hello World! More text should be here")
            # assert updatetextwindow is not None, "Window was none"
            # self.viz.text("on cluster", win=updatetextwindow, append=True)

    def plot(self, stats, epoch):
        for k, v in stats.items():
            win_id = self.ip_ts_desc + k[:3]

            if isinstance(v, (torch.Tensor, np.ndarray)):
                self.viz.images(
                    v,
                    win=win_id,
                    # nrow=4,
                    env=self.env,
                    opts=dict(
                        title=self.ip_ts_desc,
                        caption=k,
                        store_history=True,
                    ),
                )

            else:
                # print("key", k[:-6])
                # print("win id", win_id)
                if win_id not in self.plots:  # e.g. no loss plot yet
                    self.plots[win_id] = self.viz.line(
                        #     self.viz.line(
                        X=[epoch],
                        Y=[v],
                        env=self.env,
                        ## win=self.plots[win_id],
                        # win=win_id,
                        # name=k[:-6],
                        # update="append",
                        opts=dict(
                            # legend=[k[:-5]],
                            legend=[k],
                            title=self.ip_ts_desc,
                            xlabel="Epochs",
                            ylabel=k[:3],
                            name=k,
                            # fillarea=True,
                            # showlegend=True,
                            # width=250,
                            # height=250,
                            # ytype="log",
                            # marginleft=30,
                            # marginright=30,
                            # marginbottom=80,
                            # margintop=30,
                        ),
                    )
                else:
                    self.viz.line(
                        # X=np.array([epoch]),
                        # Y=np.array([v]),
                        X=[epoch],
                        Y=[v],
                        env=self.env,
                        win=self.plots[win_id],
                        # name=k[:-5],
                        name=k,
                        update="append",
                    )
        stats.clear()

if __name__ == "__main__":
    VisdomPlotter()
