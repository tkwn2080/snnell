# This will implement initially a plain novelty search: squared euclidean distance from the emitter
# It will create an archive and compare the entity to the population and the archive
# It will return the top n least similar entities
# How is this to interact with fitness? Both will be reported, but for now only novelty used
# We should perhaps amalgamate the two into a single class called Selection