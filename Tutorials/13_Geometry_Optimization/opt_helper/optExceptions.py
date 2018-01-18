
# We don't catch this one internally.
class OPT_FAIL(Exception):
    def __init__(self, mesg='None given'):
        print('OPT_FAIL: Optimization has failed.')
        #Exception.__init__(self, mesg)


class ALG_FAIL(Exception):
    #maybe generalize later def __init__(self, *args, **kwargs):
    def __init__(self, mesg='None given', newLinearBends=None):
        print('ALG_FAIL: Exception created.\n')
        if newLinearBends:
            print('ALG_FAIL: New bends detected.\n')
        self.linearBends = newLinearBends
        self.mesg = mesg
